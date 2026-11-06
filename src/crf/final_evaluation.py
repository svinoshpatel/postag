"""Frozen three-role evaluation with an immutable, validated inference cache."""

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, BinaryIO

from crf.data.conllu import GoldSentence, iter_gold_records, iter_gold_sentences
from crf.data.identity import iter_input_identities
from crf.data.records import SentenceRecord
from crf.evaluation import (
    GoldVocabulary,
    ModelReference,
    SentencePrediction,
    gold_vocabulary,
    score_test,
    validate_metric_report,
)
from crf.experiments import FrozenExperiment, file_sha256, load_development, read_bytes
from crf.models.bundles import _open_file, load_bundle
from crf.models.confidence import score_sentence
from crf.pipeline import _context, _publish, _utc_now
from crf.predictions import Prediction, parse_prediction
from crf.reporting import validate_observation
from crf.utils.artifacts import _parent_fd, atomic_writer
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    jsonl_bytes,
    loads_json,
    object_fields,
)

ROLES = ("gate", "final", "gold_control")
CORE_OUTPUTS = tuple(f"{role}.jsonl" for role in ROLES) + ("metrics.json",)


@dataclass(frozen=True)
class EvaluationCore:
    experiment: FrozenExperiment
    directory: Path
    metrics: dict[str, object]
    gold: tuple[GoldSentence, ...]


def _inputs(
    experiment: FrozenExperiment, root: Path, monitor: MemoryMonitor
) -> tuple[tuple[SentenceRecord, ...], tuple[GoldSentence, ...], GoldVocabulary]:
    config = experiment.config
    inputs = tuple(iter_input_identities(config, root, monitor=monitor))
    train = tuple(iter_gold_records(inputs[0], config, root, monitor=monitor))
    gold = tuple(
        iter_gold_sentences(
            inputs[2], config, root, final_evaluation=True, monitor=monitor
        )
    )
    return (
        tuple(sentence.to_record(config, final_evaluation=True) for sentence in gold),
        gold,
        gold_vocabulary(train, config),
    )


def cached_predictions(
    path: Path,
    records: tuple[SentenceRecord, ...],
    experiment: FrozenExperiment,
    reference: ModelReference,
    monitor: MemoryMonitor,
) -> Iterator[SentencePrediction]:
    with _open_file(path.parent, path.name) as stream:
        for record, line in zip_longest(records, stream):
            monitor.checkpoint()
            if record is None or line is None:
                raise ValidationError("Missing/extra cached prediction")
            prediction = parse_prediction(loads_json(line), experiment.config)
            expected = (
                experiment.run_id,
                reference,
                record.sentence_id,
                record.source_path,
                record.source_sha256,
                "test_gold",
                record.sentence_ordinal,
                record.token_ids,
                record.original_forms,
                record.normalized_forms,
                record.token_spans,
            )
            actual = (
                prediction.run_id,
                prediction.model_ref,
                prediction.sentence_id,
                prediction.source_path,
                prediction.source_sha256,
                prediction.source_role,
                prediction.sentence_ordinal,
                prediction.token_ids,
                prediction.original_forms,
                prediction.normalized_forms,
                prediction.token_spans,
            )
            if actual != expected:
                raise ValidationError("Cached prediction alignment/provenance mismatch")
            yield SentencePrediction(
                prediction.sentence_id,
                prediction.token_ids,
                prediction.normalized_forms,
                prediction.predicted_labels,
            )


def _load_core(experiment: FrozenExperiment, root: Path) -> EvaluationCore:
    directory = root / "reports" / experiment.run_id / "evaluation"
    raw = object_fields(
        loads_json(read_bytes(directory / "core.json")),
        {
            "schema_version",
            "run_id",
            "status",
            "development_sha256",
            "selected_sha256",
            "outputs",
            "started_at",
            "finished_at",
            "environment",
            "peak_rss_mib",
        },
        "evaluation core",
    )
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["run_id"] != experiment.run_id
        or raw["status"] != "predictions_complete"
    ):
        raise ValidationError("Invalid evaluation core")
    validate_observation(
        raw["environment"], raw["started_at"], raw["finished_at"], raw["peak_rss_mib"]
    )
    outputs = object_fields(raw["outputs"], set(CORE_OUTPUTS), "core outputs")
    with MemoryMonitor(experiment.config.resources.memory_budget_mib) as monitor:
        if raw["development_sha256"] != file_sha256(
            experiment.run_path, monitor
        ) or raw["selected_sha256"] != file_sha256(
            root / "models" / experiment.run_id / "selected.json", monitor
        ):
            raise ValidationError("Evaluation core frozen-history mismatch")
        for name, checksum in outputs.items():
            if checksum != file_sha256(directory / name, monitor):
                raise ValidationError("Evaluation core output checksum mismatch")
        records, gold, vocabulary = _inputs(experiment, root, monitor)
        report = validate_metric_report(
            loads_json(read_bytes(directory / "metrics.json")),
            experiment.config,
            split="test_gold",
        )
        expected_models = {}
        for role in ROLES:
            reference = getattr(experiment.selection, role)
            expected_models[role] = score_test(
                records,
                cached_predictions(
                    directory / f"{role}.jsonl", records, experiment, reference, monitor
                ),
                config=experiment.config,
                vocabulary=vocabulary,
                model_ref=reference,
                final_evaluation=True,
            )
        if report != dict(
            schema_version=1,
            run_id=experiment.run_id,
            split="test_gold",
            target_inventory=list(experiment.config.target_inventory),
            models=expected_models,
        ):
            raise ValidationError(
                "Cached test metrics disagree with aligned predictions"
            )
    monitor.ensure_success()
    return EvaluationCore(experiment, directory, report, gold)


def evaluate_core(
    root: Path, run_id: str, *, seed: int | None = None
) -> EvaluationCore:
    root = root.resolve()
    experiment = load_development(root, run_id)
    config = experiment.config
    if seed is not None and (
        type(seed) is not int or seed != config.evaluation.bootstrap_seed
    ):
        raise ValidationError("Evaluation seed must equal frozen bootstrap seed")
    directory = root / "reports" / run_id / "evaluation"
    if directory.exists() or directory.is_symlink():
        return _load_core(experiment, root)
    context = _context(root)
    with _parent_fd(directory.relative_to(root), root, create=True) as parent:
        os.mkdir(directory.name, mode=0o700, dir_fd=parent)
    with MemoryMonitor(config.resources.memory_budget_mib) as outer:
        records, gold, vocabulary = _inputs(experiment, root, outer)
        models = {}
        for role in ROLES:
            reference = getattr(experiment.selection, role)
            loaded = load_bundle(
                root / "models" / run_id / "candidates" / reference.candidate_id,
                repository_root=root,
                expected_run_id=run_id,
                expected_candidate_id=reference.candidate_id,
                expected_model_sha256=reference.model_sha256,
                expected_metadata_sha256=reference.metadata_sha256,
                monitor=outer,
            )
            digest = hashlib.sha256()
            inference = MemoryMonitor(config.resources.memory_budget_mib)

            def validate(
                stream: BinaryIO,
                successful: MemoryMonitor = inference,
                expected: Callable[[], bytes] = digest.digest,
            ) -> None:
                successful.ensure_success()
                actual = hashlib.sha256()
                while chunk := stream.read(config.resources.read_buffer_bytes):
                    actual.update(chunk)
                if actual.digest() != expected():
                    raise ValidationError("Prediction publication changed")

            try:
                with atomic_writer(
                    directory / f"{role}.jsonl", root, validate=validate
                ) as stream:
                    with inference:

                        def predictions(
                            model: Any = loaded.model,
                            active: MemoryMonitor = inference,
                            ref: ModelReference = reference,
                            update: Callable[[bytes], None] = digest.update,
                        ) -> Iterator[SentencePrediction]:
                            for record in records:
                                confidence = score_sentence(
                                    record.feature_forms,
                                    model=model,
                                    config=config,
                                    monitor=active,
                                )
                                prediction = Prediction(
                                    1,
                                    run_id,
                                    ref,
                                    record.sentence_id,
                                    record.source_path,
                                    record.source_sha256,
                                    "test_gold",
                                    record.sentence_ordinal,
                                    record.token_ids,
                                    record.original_forms,
                                    record.normalized_forms,
                                    record.token_spans,
                                    confidence.labels,
                                    confidence.token_confidences,
                                )
                                encoded = jsonl_bytes(prediction.to_dict())
                                stream.write(encoded)
                                update(encoded)
                                yield SentencePrediction(
                                    record.sentence_id,
                                    record.token_ids,
                                    record.normalized_forms,
                                    confidence.labels,
                                )

                        models[role] = score_test(
                            records,
                            predictions(),
                            config=config,
                            vocabulary=vocabulary,
                            model_ref=reference,
                            final_evaluation=True,
                        )
            finally:
                loaded.close()
        from crf.utils.serialization import json_bytes

        report = dict(
            schema_version=1,
            run_id=run_id,
            split="test_gold",
            target_inventory=list(config.target_inventory),
            models=models,
        )
        validate_metric_report(report, config, split="test_gold")
        _publish(directory / "metrics.json", json_bytes(report), root)
        if load_development(root, run_id) != experiment:
            raise ValidationError("Frozen history changed during evaluation")
    outer.ensure_success()
    core = dict(
        schema_version=1,
        run_id=run_id,
        status="predictions_complete",
        started_at=context.started_at,
        finished_at=_utc_now(),
        development_sha256=file_sha256(experiment.run_path),
        selected_sha256=file_sha256(root / "models" / run_id / "selected.json"),
        outputs={name: file_sha256(directory / name) for name in CORE_OUTPUTS},
        environment=dict(
            python_version=context.python_version,
            unicode_version=context.unicode_version,
            dependency_versions=dict(context.dependency_versions),
            source_revision=context.source_revision,
            source_dirty=context.source_dirty,
        ),
        peak_rss_mib=outer.peak_rss_mib,
    )
    _publish(directory / "core.json", json_bytes(core), root)
    return _load_core(experiment, root)
