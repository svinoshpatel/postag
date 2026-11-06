"""Development search orchestration; frozen experiment authorization is separate."""

import hashlib
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, cast

from crf.data.conllu import iter_gold_records
from crf.data.contamination import (
    EligibleSentence,
    KnownSourceExclusion,
    contamination_audit,
)
from crf.data.identity import InputIdentity, iter_input_identities
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.evaluation import (
    GoldVocabulary,
    ModelReference,
    SentencePrediction,
    gold_vocabulary,
    publish_validation_report,
    score_validation,
)
from crf.features.surface import sentence_features
from crf.models.bundles import _dispose, _open_file, load_bundle, save_bundle
from crf.models.confidence import iter_scored_candidates
from crf.models.reuse import reuse_empty_selections
from crf.models.sampling import FrozenDataset, freeze_datasets
from crf.models.search import BaselineSearch, BaselineTrial, search_baseline
from crf.models.training import candidate_parameters, fit_candidate
from crf.utils.artifacts import allocate_run, publish_bytes
from crf.utils.config import PipelineConfig, parse_config, validate_input_paths
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    canonical_config_bytes,
    jsonl_bytes,
    loads_json,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalTrial:
    threshold: float
    outcome: BaselineTrial
    reused: bool = False

    def to_dict(self) -> dict[str, object]:
        result = self.outcome.to_dict()
        result.update(stage="final", threshold=self.threshold)
        if self.reused:
            result["status"] = "reused"
        return result


@dataclass(frozen=True)
class DevelopmentSearch:
    run_id: str
    baseline: BaselineSearch
    datasets: tuple[FrozenDataset, ...]
    final_trials: tuple[FinalTrial, ...]
    trials_path: Path


class DevelopmentIncomplete(RuntimeError):
    def __init__(self, run_id: str, stage: str, message: str) -> None:
        super().__init__(f"{run_id}: {stage}: {message}")
        self.run_id = run_id
        self.stage = stage


def _publish(path: Path, content: bytes, root: Path) -> None:
    def validate(stream: BinaryIO) -> None:
        if stream.read() != content:
            raise ValidationError("Development publication changed")

    publish_bytes(path, content, root, validate=validate)


def _dataset_records(
    dataset: FrozenDataset,
    sources: dict[str, InputIdentity],
    config: PipelineConfig,
    monitor: MemoryMonitor,
) -> Iterator[SentenceRecord]:
    digest = hashlib.sha256()
    gold_tokens = pseudo_tokens = pseudo_sentences = 0
    seen_pseudo = False
    with _open_file(dataset.path.parent, dataset.path.name) as stream:
        for line in stream:
            monitor.checkpoint()
            digest.update(line)
            value = loads_json(line)
            if not isinstance(value, dict) or value.get("source_path") not in sources:
                raise ValidationError("Dataset source outside frozen inputs")
            source = sources[cast(str, value["source_path"])]
            record = parse_sentence_record(
                value,
                config=config,
                source=source,
                decoded_source_length=source.size_bytes,
            )
            if record.record_type == "gold":
                if source.source_role != "train_gold" or seen_pseudo:
                    raise ValidationError(
                        "Final dataset requires gold first, without held-out labels"
                    )
                gold_tokens += len(record.token_ids)
            else:
                seen_pseudo = True
                pseudo_sentences += 1
                pseudo_tokens += len(record.token_ids)
            yield record
    if (
        digest.hexdigest() != dataset.sha256
        or gold_tokens != dataset.gold_tokens
        or pseudo_tokens != dataset.pseudo_tokens
        or pseudo_sentences != dataset.pseudo_sentences
    ):
        raise ValidationError("Final training dataset identity/counts changed")


def _fit_final(
    dataset: FrozenDataset,
    trial: BaselineTrial,
    validation: tuple[SentenceRecord, ...],
    vocabulary: GoldVocabulary,
    sources: dict[str, InputIdentity],
    config: PipelineConfig,
    root: Path,
) -> BaselineTrial:
    model = None
    try:
        parameters = dict(trial.crf_parameters)
        with MemoryMonitor(config.resources.memory_budget_mib) as training:
            candidate = fit_candidate(
                _dataset_records(dataset, sources, config, training),
                config=config,
                c1=cast(float, parameters["c1"]),
                c2=cast(float, parameters["c2"]),
                allow_pseudo=True,
                monitor=training,
            )
            model = candidate.model
        training.ensure_success()
        if candidate.record.training_dataset_sha256 != dataset.sha256:
            raise ValidationError("Final fit dataset checksum mismatch")
        metadata = save_bundle(
            candidate,
            config=config,
            repository_root=root,
            run_id=trial.run_id,
            candidate_id=trial.trial_id,
            monitor=training,
        )
        with _open_file(
            root / "models" / trial.run_id / "candidates" / trial.trial_id,
            "metadata.json",
        ) as stream:
            sidecar_sha = hashlib.sha256(stream.read()).hexdigest()
        reference = ModelReference(trial.trial_id, metadata.model_sha256, sidecar_sha)
        with MemoryMonitor(config.resources.memory_budget_mib) as inference:

            def predictions() -> Iterable[SentencePrediction]:
                for record in validation:
                    inference.checkpoint()
                    features = sentence_features(
                        record.feature_forms,
                        schema_version=config.features.schema_version,
                        normalizer_version=config.preprocessing.normalizer_version,
                    )
                    yield SentencePrediction(
                        record.sentence_id,
                        record.token_ids,
                        record.feature_forms,
                        tuple(model.predict_single(features)),
                    )

            metrics = score_validation(
                validation,
                predictions(),
                config=config,
                vocabulary=vocabulary,
                model_ref=reference,
            )
        inference.ensure_success()
        publish_validation_report(
            metrics,
            config=config,
            repository_root=root,
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            monitor=inference,
        )
        return BaselineTrial(
            trial.run_id,
            trial.trial_id,
            trial.crf_parameters,
            dataset.sha256,
            "fitted",
            reference,
            cast(float, metrics["accuracy"]),
            cast(float, metrics["macro_f1"]),
            None,
        )
    finally:
        if model is not None:
            _dispose(model)


def develop(
    config: PipelineConfig,
    repository_root: Path,
    *,
    known_source_exclusions: Sequence[KnownSourceExclusion] = (),
) -> DevelopmentSearch:
    """Run one fresh search; no selected models/completion descriptor is published."""
    root = repository_root.resolve()
    if (
        parse_config(config.to_dict(), repository_root=root, resolve_paths=False)
        != config
    ):
        raise ValidationError("Noncanonical development configuration")
    paths = allocate_run(root)
    run_id = paths.run_id
    grid = tuple(
        sorted(
            (c1, c2) for c1 in config.training.c1_grid for c2 in config.training.c2_grid
        )
    )

    def pending(
        trial_id: str, c1: float, c2: float, reason: str = "Search not reached"
    ) -> BaselineTrial:
        return BaselineTrial(
            run_id,
            trial_id,
            tuple(sorted(candidate_parameters(config, c1, c2).items())),
            None,
            "not_executed",
            None,
            None,
            None,
            ("not_executed", reason),
        )

    baseline_trials = tuple(
        pending(f"baseline_{i:03d}", c1, c2) for i, (c1, c2) in enumerate(grid)
    )
    finals = [
        FinalTrial(threshold, pending(f"final_{i:03d}_{j:03d}", c1, c2))
        for i, threshold in enumerate(sorted(config.self_training.thresholds))
        for j, (c1, c2) in enumerate(grid)
    ]

    def capture(trials: tuple[BaselineTrial, ...]) -> None:
        nonlocal baseline_trials
        baseline_trials = trials

    failure: BaseException | None = None
    stage = "prepare"
    baseline: BaselineSearch | None = None
    datasets: tuple[FrozenDataset, ...] = ()
    try:
        logger.info("Starting development search %s", run_id)
        with MemoryMonitor(config.resources.memory_budget_mib) as preparation:
            validate_input_paths(config, root)
            inputs = tuple(iter_input_identities(config, root, monitor=preparation))
            sources = {source.source_path: source for source in inputs}
            train = tuple(
                iter_gold_records(inputs[0], config, root, monitor=preparation)
            )
            validation = tuple(
                iter_gold_records(inputs[1], config, root, monitor=preparation)
            )
            vocabulary = gold_vocabulary(train, config)
        preparation.ensure_success()
        _publish(
            paths.reports / "effective_config.json",
            canonical_config_bytes(config),
            root,
        )
        _publish(
            paths.reports / "inputs.jsonl",
            b"".join(jsonl_bytes(source.to_dict()) for source in inputs),
            root,
        )
        stage = "baseline"
        baseline = search_baseline(
            train,
            validation,
            config=config,
            repository_root=root,
            run_id=run_id,
            on_trials=capture,
        )
        stage = "pseudo_label"
        gate = baseline.gate
        with MemoryMonitor(config.resources.memory_budget_mib) as scoring:
            loaded = load_bundle(
                root / "models" / run_id / "candidates" / gate.candidate_id,
                repository_root=root,
                expected_run_id=run_id,
                expected_candidate_id=gate.candidate_id,
                expected_model_sha256=gate.model_sha256,
                expected_metadata_sha256=gate.metadata_sha256,
                monitor=scoring,
            )
            try:
                if loaded.metadata.config != config:
                    raise ValidationError("Gate configuration mismatch")
                with contamination_audit(
                    config,
                    root,
                    index_path=Path("data/interim") / run_id / "contamination.sqlite3",
                    known_source_exclusions=known_source_exclusions,
                    monitor=scoring,
                ) as audit:
                    if audit.inputs != inputs:
                        raise ValidationError(
                            "Inputs changed before pseudo-label inference"
                        )
                    sentences = (
                        event.sentence
                        for event in audit.iter_events()
                        if isinstance(event, EligibleSentence)
                    )
                    datasets = freeze_datasets(
                        train,
                        iter_scored_candidates(
                            sentences,
                            model=loaded.model,
                            config=config,
                            monitor=scoring,
                        ),
                        config=config,
                        repository_root=root,
                        run_id=run_id,
                        gate_model_sha256=gate.model_sha256,
                        all_scored=True,
                        audit_report=audit.report,
                    )
            finally:
                loaded.close()
        scoring.ensure_success()
        finals = [
            FinalTrial(
                f.threshold,
                replace(
                    f.outcome, training_dataset_sha256=datasets[i // len(grid)].sha256
                ),
            )
            for i, f in enumerate(finals)
        ]
        stage = "reuse"
        empty = reuse_empty_selections(
            datasets, baseline, config=config, repository_root=root, run_id=run_id
        )
        cached = {trial.trial_id: trial for trial in empty.trials}
        for i, final in enumerate(finals):
            dataset = next(d for d in datasets if d.threshold == final.threshold)
            if final.outcome.trial_id in cached:
                trial = cached[final.outcome.trial_id]
                original = trial.baseline
                # Reused model keeps its original training history and metrics.
                finals[i] = FinalTrial(
                    final.threshold,
                    BaselineTrial(
                        run_id,
                        trial.trial_id,
                        original.crf_parameters,
                        original.training_dataset_sha256,
                        "fitted",
                        original.model_ref,
                        original.validation_accuracy,
                        original.validation_macro_f1,
                        None,
                    ),
                    reused=True,
                )
        stage = "final"
        for i, final in enumerate(finals):
            if final.outcome.trial_id in cached:
                continue
            dataset = next(d for d in datasets if d.threshold == final.threshold)
            try:
                outcome = _fit_final(
                    dataset,
                    final.outcome,
                    validation,
                    vocabulary,
                    sources,
                    config,
                    root,
                )
                finals[i] = FinalTrial(final.threshold, outcome)
            except BaseException as exc:
                finals[i] = FinalTrial(
                    final.threshold,
                    BaselineTrial(
                        run_id,
                        final.outcome.trial_id,
                        final.outcome.crf_parameters,
                        dataset.sha256,
                        "failed",
                        None,
                        None,
                        None,
                        ("final", str(exc) or type(exc).__name__),
                    ),
                )
                raise
        stage = "input_verification"
        with MemoryMonitor(config.resources.memory_budget_mib) as verification:
            if (
                tuple(iter_input_identities(config, root, monitor=verification))
                != inputs
            ):
                raise ValidationError("Inputs changed during development search")
        verification.ensure_success()
    except BaseException as exc:
        failure = exc
    # Publish every planned outcome, including those never reached after abort.
    content_rows = [trial.to_dict() for trial in baseline_trials]
    for final in finals:
        raw = final.to_dict()
        if raw["status"] == "not_executed":
            raw["error"] = dict(
                stage="not_executed",
                message=f"{stage}: {failure or 'Search not reached'}",
            )
        content_rows.append(raw)
    ledger = paths.reports / "trials.jsonl"
    _publish(ledger, b"".join(jsonl_bytes(row) for row in content_rows), root)
    if failure is not None:
        raise DevelopmentIncomplete(
            run_id, stage, str(failure) or type(failure).__name__
        ) from failure
    assert baseline is not None
    return DevelopmentSearch(run_id, baseline, datasets, tuple(finals), ledger)
