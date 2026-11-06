"""Strict saved-search validation, model selection and development authorization."""

import hashlib
import math
import platform
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal, cast

from crf.data.contamination import sentence_fingerprint
from crf.data.identity import InputIdentity, iter_input_identities, parse_input_identity
from crf.data.records import parse_sentence_record
from crf.evaluation import ModelReference, validate_metric_report
from crf.models.bundles import BundleMetadata, _open_file, parse_metadata
from crf.models.training import (
    ParameterValue,
    candidate_parameters,
    dependency_versions,
)
from crf.utils.artifacts import load_completed
from crf.utils.config import PipelineConfig, parse_config
from crf.utils.resources import MemoryMonitor, check_memory
from crf.utils.serialization import (
    ValidationError,
    canonical_config_bytes,
    config_sha256,
    integer,
    json_bytes,
    jsonl_bytes,
    loads_json,
    object_fields,
    probability,
    sha256_value,
    text_value,
)


@dataclass(frozen=True)
class Trial:
    run_id: str
    trial_id: str
    stage: Literal["baseline", "final"]
    threshold: float | None
    parameters: tuple[tuple[str, ParameterValue], ...]
    dataset_sha256: str | None
    status: Literal["fitted", "reused", "failed", "not_executed"]
    model: ModelReference | None
    accuracy: float | None
    macro_f1: float | None
    error: tuple[str, str] | None


@dataclass(frozen=True)
class Selection:
    run_id: str
    effective_config_sha256: str
    selected_threshold: float
    gate: ModelReference
    final: ModelReference
    gold_control: ModelReference

    def to_dict(self) -> dict[str, object]:
        return dict(
            schema_version=1,
            run_id=self.run_id,
            effective_config_sha256=self.effective_config_sha256,
            selected_threshold=self.selected_threshold,
            gate=self.gate.to_dict(),
            final=self.final.to_dict(),
            gold_control=self.gold_control.to_dict(),
        )


@dataclass(frozen=True)
class DatasetStats:
    sha256: str
    gold_sha256: str
    gold_sentences: int
    gold_tokens: int
    pseudo_sentences: int
    pseudo_tokens: int


@dataclass(frozen=True)
class FrozenExperiment:
    run_id: str
    config: PipelineConfig
    selection: Selection
    run_path: Path


TRIAL_FIELDS = {
    "schema_version",
    "run_id",
    "trial_id",
    "stage",
    "threshold",
    "crf_parameters",
    "training_dataset_sha256",
    "status",
    "model_ref",
    "validation_accuracy",
    "validation_macro_f1",
    "error",
}
RUN_FIELDS = {
    "schema_version",
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "effective_config_sha256",
    "inputs_sha256",
    "trials_sha256",
    "selected_sha256",
    "source_revision",
    "source_dirty",
    "python_version",
    "unicode_version",
    "dependency_versions",
    "peak_rss_mib",
    "error",
}


def identifier(value: object) -> str:
    result = text_value(value, "identifier")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result):
        raise ValidationError("Invalid identifier")
    return result


def _reference(value: object) -> ModelReference:
    raw = object_fields(
        value, {"candidate_id", "model_sha256", "metadata_sha256"}, "model_ref"
    )
    return ModelReference(
        identifier(raw["candidate_id"]),
        sha256_value(raw["model_sha256"], "model checksum"),
        sha256_value(raw["metadata_sha256"], "metadata checksum"),
    )


def parse_trial(value: object) -> Trial:
    raw = object_fields(value, TRIAL_FIELDS, "trial")
    if integer(raw["schema_version"], "schema_version") != 1 or raw["stage"] not in (
        "baseline",
        "final",
    ):
        raise ValidationError("Invalid trial stage/schema")
    stage: Literal["baseline", "final"] = raw["stage"]
    threshold = (
        None if stage == "baseline" else probability(raw["threshold"], "threshold")
    )
    if stage == "baseline" and raw["threshold"] is not None:
        raise ValidationError("Baseline threshold must be null")
    status = raw["status"]
    if status not in ("fitted", "reused", "failed", "not_executed"):
        raise ValidationError("Invalid trial status")
    parameters = raw["crf_parameters"]
    if not isinstance(parameters, dict) or any(type(k) is not str for k in parameters):
        raise ValidationError("Invalid trial parameters")
    # JSON validation rejects nonfinite/unsupported values; grid comparison binds types.
    json_bytes(parameters)
    dataset = (
        None
        if raw["training_dataset_sha256"] is None
        else sha256_value(raw["training_dataset_sha256"], "dataset checksum")
    )
    model = None if raw["model_ref"] is None else _reference(raw["model_ref"])
    accuracy = (
        None
        if raw["validation_accuracy"] is None
        else probability(raw["validation_accuracy"], "accuracy")
    )
    f1 = (
        None
        if raw["validation_macro_f1"] is None
        else probability(raw["validation_macro_f1"], "macro_f1")
    )
    error = None
    if raw["error"] is not None:
        error_raw = object_fields(raw["error"], {"stage", "message"}, "error")
        error = (
            text_value(error_raw["stage"], "error stage"),
            text_value(error_raw["message"], "error message"),
        )
    if status in ("fitted", "reused"):
        if (
            model is None
            or dataset is None
            or accuracy is None
            or f1 is None
            or error is not None
        ):
            raise ValidationError("Incomplete successful trial")
    elif model is not None or accuracy is not None or f1 is not None or error is None:
        raise ValidationError("Invalid unsuccessful trial")
    return Trial(
        identifier(raw["run_id"]),
        identifier(raw["trial_id"]),
        stage,
        threshold,
        tuple(sorted(cast(dict[str, ParameterValue], parameters).items())),
        dataset,
        status,
        model,
        accuracy,
        f1,
        error,
    )


def parse_selection(value: object, config: PipelineConfig, run_id: str) -> Selection:
    raw = object_fields(
        value,
        {
            "schema_version",
            "run_id",
            "effective_config_sha256",
            "selected_threshold",
            "gate",
            "final",
            "gold_control",
        },
        "selection",
    )
    if integer(raw["schema_version"], "schema_version") != 1 or raw["run_id"] != run_id:
        raise ValidationError("Selection identity mismatch")
    threshold = probability(raw["selected_threshold"], "selected_threshold")
    if threshold not in config.self_training.thresholds or raw[
        "effective_config_sha256"
    ] != config_sha256(config):
        raise ValidationError("Selection configuration mismatch")
    return Selection(
        run_id,
        config_sha256(config),
        threshold,
        _reference(raw["gate"]),
        _reference(raw["final"]),
        _reference(raw["gold_control"]),
    )


def read_bytes(path: Path) -> bytes:
    with _open_file(path.parent, path.name) as stream:
        return stream.read()


def file_sha256(path: Path, monitor: MemoryMonitor | None = None) -> str:
    digest = hashlib.sha256()
    with _open_file(path.parent, path.name) as stream:
        while chunk := stream.read(65536):
            check_memory(monitor)
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(
    path: Path,
    sources: dict[str, InputIdentity],
    config: PipelineConfig,
    threshold: float | None,
    gate_sha: str | None,
    monitor: MemoryMonitor | None,
) -> DatasetStats:
    digest, gold_digest = hashlib.sha256(), hashlib.sha256()
    gold_sentences = gold_tokens = pseudo_sentences = pseudo_tokens = 0
    gold_ids: set[str] = set()
    previous_priority: tuple[str, str] | None = None
    with _open_file(path.parent, path.name) as stream:
        for line in stream:
            check_memory(monitor)
            digest.update(line)
            value = loads_json(line)
            if (
                not isinstance(value, dict)
                or type(value.get("source_path")) is not str
                or value["source_path"] not in sources
            ):
                raise ValidationError("Training source outside recorded inputs")
            source = sources[cast(str, value["source_path"])]
            record = parse_sentence_record(
                value,
                config=config,
                source=source,
                decoded_source_length=source.size_bytes,
            )
            if jsonl_bytes(record.to_dict()) != line:
                raise ValidationError(
                    "Training dataset must use canonical record bytes"
                )
            if record.record_type == "gold":
                if (
                    source.source_role != "train_gold"
                    or pseudo_sentences
                    or record.sentence_id in gold_ids
                ):
                    raise ValidationError("Invalid gold ordering, role or duplicate")
                gold_ids.add(record.sentence_id)
                gold_digest.update(line)
                gold_sentences += 1
                gold_tokens += len(record.token_ids)
            else:
                pseudo = record.pseudo
                assert pseudo is not None
                if (
                    threshold is None
                    or pseudo.min_confidence < threshold
                    or pseudo.gate_model_sha256 != gate_sha
                ):
                    raise ValidationError("Pseudo confidence/gate mismatch")
                fingerprint = sentence_fingerprint(
                    record.surface_text,
                    normalizer_version=config.preprocessing.normalizer_version,
                )
                priority = (pseudo.sampling_priority, fingerprint)
                if fingerprint != pseudo.sentence_fingerprint or (
                    previous_priority is not None and priority <= previous_priority
                ):
                    raise ValidationError(
                        "Invalid pseudo fingerprint/priority ordering"
                    )
                previous_priority = priority
                pseudo_sentences += 1
                pseudo_tokens += len(record.token_ids)
    if not gold_tokens:
        raise ValidationError("Empty supervised dataset")
    return DatasetStats(
        digest.hexdigest(),
        gold_digest.hexdigest(),
        gold_sentences,
        gold_tokens,
        pseudo_sentences,
        pseudo_tokens,
    )


def _metrics(trial: Trial, config: PipelineConfig, root: Path) -> None:
    assert trial.model is not None
    report_id = trial.model.candidate_id
    raw = validate_metric_report(
        loads_json(
            read_bytes(
                root
                / "reports"
                / trial.run_id
                / "validation"
                / report_id
                / "metrics.json"
            )
        ),
        config,
    )
    models = cast(dict[str, dict[str, object]], raw["models"])
    if raw["run_id"] != trial.run_id or set(models) != {report_id}:
        raise ValidationError("Cached validation identity mismatch")
    metrics = models[report_id]
    if (
        metrics["model_ref"] != trial.model.to_dict()
        or metrics["accuracy"] != trial.accuracy
        or metrics["macro_f1"] != trial.macro_f1
    ):
        raise ValidationError("Cached validation result mismatch")


def _bundle(
    trial: Trial,
    dataset: DatasetStats,
    config: PipelineConfig,
    root: Path,
    monitor: MemoryMonitor | None,
    cache: dict[ModelReference, BundleMetadata],
) -> None:
    assert trial.model is not None
    reference = trial.model
    if reference not in cache:
        directory = (
            root / "models" / trial.run_id / "candidates" / reference.candidate_id
        )
        encoded = read_bytes(directory / "metadata.json")
        if hashlib.sha256(encoded).hexdigest() != reference.metadata_sha256:
            raise ValidationError("Bundle metadata checksum mismatch")
        metadata = parse_metadata(loads_json(encoded))
        if (
            metadata.run_id != trial.run_id
            or metadata.candidate_id != reference.candidate_id
            or metadata.config != config
        ):
            raise ValidationError("Bundle configuration/identity mismatch")
        if (
            metadata.model_sha256 != reference.model_sha256
            or file_sha256(directory / "model.joblib", monitor)
            != reference.model_sha256
        ):
            raise ValidationError("Bundle model checksum mismatch")
        cache[reference] = metadata
    training = cache[reference].training
    if (
        training.training_dataset_sha256 != trial.dataset_sha256
        or trial.dataset_sha256 != dataset.sha256
        or training.crf_parameters != trial.parameters
        or training.training_sentence_count
        != dataset.gold_sentences + dataset.pseudo_sentences
        or training.training_token_count != dataset.gold_tokens + dataset.pseudo_tokens
        or training.pseudo_sentence_count != dataset.pseudo_sentences
        or training.pseudo_token_count != dataset.pseudo_tokens
    ):
        raise ValidationError("Bundle/trial/dataset training history mismatch")
    _metrics(trial, config, root)


def final_rank(
    trial: Trial, pseudo_tokens: int
) -> tuple[float, float, int, float, float, float]:
    if trial.accuracy is None or trial.macro_f1 is None or trial.threshold is None:
        raise ValidationError("Cannot rank incomplete final trial")
    parameters = dict(trial.parameters)
    return (
        -trial.accuracy,
        -trial.macro_f1,
        pseudo_tokens,
        -trial.threshold,
        cast(float, parameters["c1"]),
        cast(float, parameters["c2"]),
    )


def validate_search(
    root: Path,
    run_id: str,
    config: PipelineConfig,
    monitor: MemoryMonitor | None = None,
) -> Selection:
    """Reconstruct selection from complete persisted evidence, never caller scores."""
    identifier(run_id)
    report_dir = root / "reports" / run_id
    if read_bytes(report_dir / "effective_config.json") != canonical_config_bytes(
        config
    ):
        raise ValidationError("Noncanonical or changed effective configuration")
    inputs = tuple(
        parse_input_identity(loads_json(line), config)
        for line in read_bytes(report_dir / "inputs.jsonl").splitlines()
    )
    if (
        len(inputs) < 4
        or tuple(source.source_role for source in inputs[:3])
        != ("train_gold", "val_gold", "test_gold")
        or any(source.source_role != "literature" for source in inputs[3:])
        or tuple(source.source_path for source in inputs[3:])
        != tuple(sorted(source.source_path for source in inputs[3:]))
        or len({source.source_path for source in inputs}) != len(inputs)
    ):
        raise ValidationError("Invalid ordered input identities")
    if tuple(iter_input_identities(config, root, monitor=monitor)) != inputs:
        raise ValidationError("Recorded input snapshot changed")
    sources = {source.source_path: source for source in inputs}
    trials = tuple(
        parse_trial(loads_json(line))
        for line in read_bytes(report_dir / "trials.jsonl").splitlines()
    )
    grid = tuple(
        sorted(
            (c1, c2) for c1 in config.training.c1_grid for c2 in config.training.c2_grid
        )
    )
    thresholds = sorted(config.self_training.thresholds)
    plan: list[tuple[str, float | None, str, float, float]] = [
        ("baseline", None, f"baseline_{j:03d}", c1, c2)
        for j, (c1, c2) in enumerate(grid)
    ]
    plan += [
        ("final", tau, f"final_{i:03d}_{j:03d}", c1, c2)
        for i, tau in enumerate(thresholds)
        for j, (c1, c2) in enumerate(grid)
    ]
    if len(trials) != len(plan):
        raise ValidationError("Incomplete development trial grid")
    for trial, (stage, tau, trial_id, c1, c2) in zip(trials, plan, strict=True):
        check_memory(monitor)
        if (
            trial.run_id != run_id
            or trial.stage != stage
            or trial.threshold != tau
            or trial.trial_id != trial_id
            or trial.status not in ("fitted", "reused")
            or json_bytes(dict(trial.parameters), sort_keys=True)
            != json_bytes(candidate_parameters(config, c1, c2), sort_keys=True)
        ):
            raise ValidationError(
                "Incomplete, failed or inconsistent development trial"
            )
        if trial.status == "fitted" and (
            trial.model is None or trial.model.candidate_id != trial.trial_id
        ):
            raise ValidationError("Fitted trial must reference its own candidate")
    baseline = trials[: len(grid)]
    if any(trial.status != "fitted" for trial in baseline):
        raise ValidationError("Baseline must contain fitted gold-only candidates")
    gate_trial = min(
        baseline,
        key=lambda t: (
            -cast(float, t.accuracy),
            -cast(float, t.macro_f1),
            cast(float, dict(t.parameters)["c1"]),
            cast(float, dict(t.parameters)["c2"]),
        ),
    )
    assert gate_trial.model is not None
    gold = _dataset(
        root / "data/processed" / run_id / "baseline_train.jsonl",
        sources,
        config,
        None,
        None,
        monitor,
    )
    cache: dict[ModelReference, BundleMetadata] = {}
    for trial in baseline:
        _bundle(trial, gold, config, root, monitor, cache)
    final = trials[len(grid) :]
    counts: dict[float, int] = {}
    for i, tau in enumerate(thresholds):
        dataset = _dataset(
            root / "data/processed" / run_id / f"extended_{i:03d}.jsonl",
            sources,
            config,
            tau,
            gate_trial.model.model_sha256,
            monitor,
        )
        cap_ratio = Fraction(str(config.self_training.pseudo_to_gold_ratio))
        cap = cap_ratio.numerator * gold.gold_tokens // cap_ratio.denominator
        if dataset.gold_sha256 != gold.sha256 or dataset.pseudo_tokens > cap:
            raise ValidationError("Augmented dataset gold prefix or cap mismatch")
        counts[tau] = dataset.pseudo_tokens
        for j, trial in enumerate(final[i * len(grid) : (i + 1) * len(grid)]):
            if dataset.pseudo_tokens:
                if trial.status != "fitted":
                    raise ValidationError("Nonempty augmentation must fit freshly")
            else:
                original = baseline[j]
                if trial.status != "reused" or (
                    trial.model,
                    trial.dataset_sha256,
                    trial.accuracy,
                    trial.macro_f1,
                ) != (
                    original.model,
                    original.dataset_sha256,
                    original.accuracy,
                    original.macro_f1,
                ):
                    raise ValidationError(
                        "Empty threshold must reuse matching baseline"
                    )
            _bundle(trial, dataset, config, root, monitor, cache)
    winner = min(final, key=lambda t: final_rank(t, counts[cast(float, t.threshold)]))
    control = next(t for t in baseline if t.parameters == winner.parameters)
    assert (
        winner.model is not None
        and control.model is not None
        and winner.threshold is not None
    )
    if tuple(iter_input_identities(config, root, monitor=monitor)) != inputs:
        raise ValidationError("Inputs changed during frozen validation")
    return Selection(
        run_id,
        config_sha256(config),
        winner.threshold,
        gate_trial.model,
        winner.model,
        control.model,
    )


def validate_run(
    value: object,
    root: Path,
    run_id: str,
    config: PipelineConfig,
    monitor: MemoryMonitor | None = None,
) -> FrozenExperiment | None:

    raw = object_fields(value, RUN_FIELDS, "development completion")
    if (
        integer(raw["schema_version"], "schema_version") != 1
        or raw["run_id"] != run_id
        or raw["status"] not in ("complete", "failed")
    ):
        raise ValidationError("Completion identity/status mismatch")
    dates = []
    for field in ("started_at", "finished_at"):
        timestamp = text_value(raw[field], field)
        if not timestamp.endswith("Z"):
            raise ValidationError("Completion timestamps require UTC Z")
        try:
            dates.append(datetime.fromisoformat(timestamp))
        except ValueError as exc:
            raise ValidationError("Invalid completion timestamp") from exc
    if dates[1] < dates[0]:
        raise ValidationError("Completion finish precedes start")
    revision = raw["source_revision"]
    if revision is not None and (
        type(revision) is not str
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)
    ):
        raise ValidationError("Invalid source revision")
    if raw["source_dirty"] is not None and type(raw["source_dirty"]) is not bool:
        raise ValidationError("Invalid source dirty state")
    if (
        raw["python_version"] != platform.python_version()
        or raw["unicode_version"] != unicodedata.unidata_version
        or raw["dependency_versions"] != dict(dependency_versions())
    ):
        raise ValidationError("Completion environment mismatch")
    peak = raw["peak_rss_mib"]
    try:
        valid_peak = (
            type(peak) in (int, float)
            and math.isfinite(cast(float, peak))
            and cast(float, peak) >= 0
        )
    except OverflowError:
        valid_peak = False
    if not valid_peak:
        raise ValidationError("Invalid peak RSS")
    if raw["effective_config_sha256"] != config_sha256(config):
        raise ValidationError("Completion configuration mismatch")
    for field in ("inputs_sha256", "trials_sha256", "selected_sha256"):
        if raw[field] is not None:
            sha256_value(raw[field], field)
    if raw["status"] == "failed":
        error = object_fields(raw["error"], {"stage", "message"}, "error")
        text_value(error["stage"], "error stage")
        text_value(error["message"], "error message")
        if raw["selected_sha256"] is not None:
            raise ValidationError("Failed development cannot authorize selection")
        return None
    if raw["error"] is not None:
        raise ValidationError("Successful completion must have null error")
    for field, path in (
        ("inputs_sha256", root / "reports" / run_id / "inputs.jsonl"),
        ("trials_sha256", root / "reports" / run_id / "trials.jsonl"),
        ("selected_sha256", root / "models" / run_id / "selected.json"),
    ):
        if raw[field] is None or raw[field] != file_sha256(path, monitor):
            raise ValidationError(f"Completion {field} mismatch")
    selection = parse_selection(
        loads_json(read_bytes(root / "models" / run_id / "selected.json")),
        config,
        run_id,
    )
    if selection != validate_search(root, run_id, config, monitor):
        raise ValidationError(
            "Frozen roles/threshold disagree with validated search ranking"
        )
    return FrozenExperiment(
        run_id, config, selection, root / "reports" / run_id / "run.json"
    )


def load_development(repository_root: Path, run_id: str) -> FrozenExperiment:
    """Authorize only a complete unchanged development run; performs no writes."""
    identifier(run_id)
    root = repository_root.resolve()
    content = read_bytes(root / "reports" / run_id / "effective_config.json")
    config = parse_config(
        loads_json(content), repository_root=root, resolve_paths=False
    )
    if canonical_config_bytes(config) != content:
        raise ValidationError("Changed/noncanonical recorded configuration")
    with MemoryMonitor(config.resources.memory_budget_mib) as monitor:
        result = load_completed(
            root / "reports" / run_id / "run.json",
            root,
            validate=lambda value: validate_run(value, root, run_id, config, monitor),
        )
    monitor.ensure_success()
    if result is None:
        raise ValidationError("Missing successful development authorization")
    return result
