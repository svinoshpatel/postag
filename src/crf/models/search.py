"""Gold-only baseline search with immutable candidate caches and terminal accounting."""

import hashlib
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

from crf.data.records import SentenceRecord
from crf.evaluation import (
    ModelReference,
    SentencePrediction,
    gold_vocabulary,
    publish_validation_report,
    score_validation,
)
from crf.features.surface import sentence_features
from crf.models.bundles import _dispose, _open_file, save_bundle
from crf.models.training import ParameterValue, candidate_parameters, fit_candidate
from crf.utils.artifacts import _parent_fd, atomic_writer, publish_bytes
from crf.utils.config import PipelineConfig, parse_config
from crf.utils.paths import validate_output_path
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    json_bytes,
    jsonl_bytes,
    loads_json,
    sha256_value,
)


@dataclass(frozen=True)
class BaselineTrial:
    run_id: str
    trial_id: str
    crf_parameters: tuple[tuple[str, ParameterValue], ...]
    training_dataset_sha256: str | None
    status: Literal["fitted", "failed", "not_executed"]
    model_ref: ModelReference | None
    validation_accuracy: float | None
    validation_macro_f1: float | None
    error: tuple[str, str] | None

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw.update(
            schema_version=1,
            stage="baseline",
            threshold=None,
            crf_parameters=dict(self.crf_parameters),
            model_ref=None if self.model_ref is None else self.model_ref.to_dict(),
            error=None
            if self.error is None
            else dict(stage=self.error[0], message=self.error[1]),
        )
        return raw


@dataclass(frozen=True)
class BaselineSearch:
    gate: ModelReference
    trials: tuple[BaselineTrial, ...]
    trials_path: Path


class SearchIncomplete(RuntimeError):
    """A full planned ledger may exist, but no gate is authorized."""

    def __init__(self, message: str, trials: tuple[BaselineTrial, ...]) -> None:
        super().__init__(message)
        self.trials = trials


def _grid(config: PipelineConfig) -> tuple[tuple[float, float], ...]:
    return tuple(
        sorted(
            (c1, c2) for c1 in config.training.c1_grid for c2 in config.training.c2_grid
        )
    )


def select_gate(
    trials: tuple[BaselineTrial, ...], config: PipelineConfig
) -> ModelReference:
    """Require complete successful grid coverage before applying frozen tie breakers."""
    grid = _grid(config)
    if (
        len(trials) != len(grid)
        or len({trial.trial_id for trial in trials}) != len(grid)
        or len({trial.run_id for trial in trials}) != 1
    ):
        raise ValidationError("Incomplete or duplicate baseline trials")
    for trial, (c1, c2) in zip(trials, grid, strict=True):
        if (
            trial.status != "fitted"
            or trial.error is not None
            or trial.model_ref is None
            or trial.training_dataset_sha256 is None
        ):
            raise ValidationError("Unsuccessful baseline search cannot select a gate")
        if json_bytes(dict(trial.crf_parameters), sort_keys=True) != json_bytes(
            candidate_parameters(config, c1, c2), sort_keys=True
        ):
            raise ValidationError("Baseline grid coverage mismatch")
        sha256_value(trial.training_dataset_sha256, "training_dataset_sha256")
        trial.model_ref.to_dict()
        for metric in (trial.validation_accuracy, trial.validation_macro_f1):
            if type(metric) not in (float, int) or not 0 <= cast(float, metric) <= 1:
                raise ValidationError("Invalid selection metric")
    if len({trial.training_dataset_sha256 for trial in trials}) != 1:
        raise ValidationError("Baseline candidates used different training datasets")
    winner = min(
        trials,
        key=lambda trial: (
            -cast(float, trial.validation_accuracy),
            -cast(float, trial.validation_macro_f1),
            dict(trial.crf_parameters)["c1"],
            dict(trial.crf_parameters)["c2"],
        ),
    )
    return cast(ModelReference, winner.model_ref)


def search_baseline(
    training_records: Iterable[SentenceRecord],
    validation_records: Iterable[SentenceRecord],
    *,
    config: PipelineConfig,
    repository_root: Path,
    run_id: str,
    on_trials: Callable[[tuple[BaselineTrial, ...]], None] | None = None,
) -> BaselineSearch:
    """Execute one fresh sequential search; only complete success returns a gate."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValidationError("Invalid baseline run ID")
    # Reject malformed directly constructed configurations before freezing the grid.
    validated = parse_config(
        config.to_dict(), repository_root=repository_root, resolve_paths=False
    )
    if validated != config:
        raise ValidationError("Noncanonical search configuration")
    root = repository_root.resolve()
    directory = root / "reports" / run_id / "baseline"
    validate_output_path(directory, root)
    with _parent_fd(directory.relative_to(root), root, create=True) as parent:
        os.mkdir(directory.name, mode=0o700, dir_fd=parent)
    grid = _grid(config)
    trials: list[BaselineTrial] = []
    dataset_sha: str | None = None
    failure: BaseException | None = None
    stage = "dataset"
    train: tuple[SentenceRecord, ...] = ()
    validation: tuple[SentenceRecord, ...] = ()
    try:
        preparation = MemoryMonitor(config.resources.memory_budget_mib)
        digest = hashlib.sha256()

        def validate_dataset(stream: BinaryIO) -> None:
            preparation.ensure_success()
            actual = hashlib.sha256()
            while chunk := stream.read(config.resources.read_buffer_bytes):
                actual.update(chunk)
            if actual.hexdigest() != digest.hexdigest():
                raise ValidationError("Published training bytes changed")

        with atomic_writer(
            root / "data/processed" / run_id / "baseline_train.jsonl",
            root,
            validate=validate_dataset,
        ) as stream:
            with preparation:

                def collect(
                    records: Iterable[SentenceRecord], role: str
                ) -> tuple[SentenceRecord, ...]:
                    result = []
                    seen = set()
                    for record in records:
                        preparation.checkpoint()
                        if (
                            record.record_type != "gold"
                            or record.source_role != role
                            or record.source_path
                            != getattr(config.data, role).as_posix()
                            or record.sentence_id in seen
                        ):
                            raise ValidationError(
                                "Baseline requires distinct declared gold train/validation records"
                            )
                        seen.add(record.sentence_id)
                        result.append(record)
                    if not result:
                        raise ValidationError("Empty baseline gold split")
                    return tuple(result)

                train = collect(training_records, "train_gold")
                validation = collect(validation_records, "val_gold")
                vocabulary = gold_vocabulary(train, config)
                for record in validation:
                    count = len(record.token_ids)
                    if (
                        not count
                        or record.original_upos is None
                        or any(
                            len(values) != count
                            for values in (
                                record.feature_forms,
                                record.target_labels,
                                record.original_upos,
                            )
                        )
                        or config.label_space.map_upos(record.original_upos)
                        != record.target_labels
                    ):
                        raise ValidationError(
                            "Invalid validation record alignment/mapping"
                        )
                for record in train:
                    preparation.checkpoint()
                    content = jsonl_bytes(record.to_dict())
                    digest.update(content)
                    stream.write(content)
            preparation.ensure_success()
        dataset_sha = digest.hexdigest()
    except BaseException as exc:
        failure = exc
    for index, (c1, c2) in enumerate(grid):
        trial_id = f"baseline_{index:03d}"
        parameters = tuple(sorted(candidate_parameters(config, c1, c2).items()))
        reference = None
        accuracy = macro_f1 = None
        error = None
        status: Literal["fitted", "failed", "not_executed"] = "not_executed"
        if failure is None:
            model = None
            try:
                stage = "fit"
                with MemoryMonitor(config.resources.memory_budget_mib) as monitor:
                    candidate = fit_candidate(
                        train, config=config, c1=c1, c2=c2, monitor=monitor
                    )
                    model = candidate.model
                monitor.ensure_success()
                if candidate.record.training_dataset_sha256 != dataset_sha:
                    raise ValidationError("Fitted dataset identity changed")
                stage = "bundle"
                metadata = save_bundle(
                    candidate,
                    config=config,
                    repository_root=root,
                    run_id=run_id,
                    candidate_id=trial_id,
                    monitor=monitor,
                )
                with _open_file(
                    root / "models" / run_id / "candidates" / trial_id, "metadata.json"
                ) as sidecar:
                    metadata_sha = hashlib.sha256(sidecar.read()).hexdigest()
                reference = ModelReference(
                    trial_id, metadata.model_sha256, metadata_sha
                )
                stage = "validation"
                with MemoryMonitor(config.resources.memory_budget_mib) as inference:

                    def predictions(
                        fitted_model: Any = model,
                    ) -> Iterable[SentencePrediction]:
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
                                tuple(fitted_model.predict_single(features)),
                            )

                    metrics = score_validation(
                        validation,
                        predictions(),
                        config=config,
                        vocabulary=vocabulary,
                        model_ref=reference,
                    )
                inference.ensure_success()
                stage = "metrics"
                publish_validation_report(
                    metrics,
                    config=config,
                    repository_root=root,
                    run_id=run_id,
                    trial_id=trial_id,
                    monitor=inference,
                )
                accuracy = cast(float, metrics["accuracy"])
                macro_f1 = cast(float, metrics["macro_f1"])
                status = "fitted"
            except BaseException as exc:
                failure = exc
                status = "failed"
                reference = None
                error = (stage, str(exc) or type(exc).__name__)
            finally:
                _dispose(model)
        else:
            error = (
                "not_executed",
                "Search aborted: " + (str(failure) or type(failure).__name__),
            )
        trials.append(
            BaselineTrial(
                run_id,
                trial_id,
                parameters,
                dataset_sha,
                status,
                reference,
                accuracy,
                macro_f1,
                error,
            )
        )
    # A preparation failure precedes every candidate; none was executed.
    ledger = tuple(trials)
    if on_trials is not None:
        on_trials(ledger)
    content = b"".join(jsonl_bytes(trial.to_dict()) for trial in ledger)
    path = directory / "trials.jsonl"

    def validate_ledger(stream: BinaryIO) -> None:
        raw = stream.read()
        expected = [trial.to_dict() for trial in ledger]
        if [loads_json(line) for line in raw.splitlines()] != expected:
            raise ValidationError("Trial ledger changed")

    publish_bytes(path, content, root, validate=validate_ledger)
    if failure is not None:
        raise SearchIncomplete("Baseline search incomplete", ledger) from failure
    return BaselineSearch(select_gate(ledger, config), ledger, path)
