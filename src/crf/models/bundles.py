"""Validated, publish-once CRF bundles. Load only trusted local joblib artifacts."""

import hashlib
import math
import os
import platform
import re
import stat
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, cast

import joblib  # type: ignore[import-untyped]
from sklearn_crfsuite import CRF  # type: ignore[import-untyped]

from crf.features.surface import sentence_features
from crf.models.training import (
    FittedCandidate,
    TrainingRecord,
    candidate_parameters,
    dependency_versions,
    optimizer_termination,
)
from crf.utils.artifacts import _parent_fd, atomic_writer, publish_bytes
from crf.utils.config import PipelineConfig, parse_config
from crf.utils.paths import reject_symlink_ancestors, validate_output_path
from crf.utils.resources import MemoryMonitor, ResourceMonitoringError, check_memory
from crf.utils.serialization import (
    ValidationError,
    config_sha256,
    integer,
    json_bytes,
    loads_json,
    object_fields,
    sha256_value,
    text_value,
)

_FIELDS = (set(TrainingRecord.__dataclass_fields__) - {"effective_config_sha256"}) | {
    "schema_version",
    "run_id",
    "candidate_id",
    "model_sha256",
    "effective_config",
}
_PROBES = (("Кіт", "спить", "."), ("Невідоме’слово", "123", "—", "Ї"), ("!",))


@dataclass(frozen=True)
class BundleMetadata:
    run_id: str
    candidate_id: str
    model_sha256: str
    config: PipelineConfig
    training: TrainingRecord

    def to_dict(self) -> dict[str, object]:
        result = asdict(self.training)
        del result["effective_config_sha256"]
        result.update(
            schema_version=1,
            run_id=self.run_id,
            candidate_id=self.candidate_id,
            model_sha256=self.model_sha256,
            effective_config=self.config.to_dict(),
            crf_parameters=dict(self.training.crf_parameters),
            dependency_versions=dict(self.training.dependency_versions),
            target_inventory=list(self.training.target_inventory),
        )
        return result


@dataclass(frozen=True)
class LoadedBundle:
    # The estimator owns a mutable native tagger and temporary model file.
    model: Any
    metadata: BundleMetadata
    metadata_sha256: str

    def close(self) -> None:
        """Dispose only dependency-owned temporary state; preserve bundle files."""
        _dispose(self.model)


def _identifier(value: object, field: str) -> str:
    result = text_value(value, field)
    if re.fullmatch(r"[A-Za-z0-9_-]+", result) is None:
        raise ValidationError(f"{field}: invalid bundle identifier")
    return result


def _equal_json(actual: object, expected: object, field: str) -> None:
    if json_bytes(actual, sort_keys=True) != json_bytes(expected, sort_keys=True):
        raise ValidationError(f"{field}: inconsistent or noncanonical values")


def _regularization(value: object) -> float:
    if type(value) not in (int, float):
        raise ValidationError("Regularization must be a finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValidationError("Regularization is too large") from exc
    if not math.isfinite(result) or result < 0:
        raise ValidationError("Regularization must be finite and nonnegative")
    return result


def parse_metadata(value: object) -> BundleMetadata:
    """Validate complete sidecar and exact environment without corpus filesystem IO."""
    raw = object_fields(value, _FIELDS, "model metadata")
    if integer(raw["schema_version"], "schema_version") != 1:
        raise ValidationError("Unsupported model schema_version")
    config = parse_config(
        raw["effective_config"], repository_root=Path(), resolve_paths=False
    )
    _equal_json(raw["effective_config"], config.to_dict(), "effective_config")
    _equal_json(
        raw["target_inventory"], list(config.target_inventory), "target_inventory"
    )
    seed = integer(raw["seed"], "seed")
    if seed != config.seed:
        raise ValidationError("seed: disagrees with effective_config")
    parameters = raw["crf_parameters"]
    if not isinstance(parameters, dict):
        raise ValidationError("crf_parameters: expected object")
    # Validate JSON before interacting with untyped estimator options.
    json_bytes(parameters)
    try:
        expected = candidate_parameters(
            config,
            _regularization(parameters.get("c1")),
            _regularization(parameters.get("c2")),
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"crf_parameters: {exc}") from exc
    _equal_json(parameters, expected, "crf_parameters")
    sentences = integer(raw["training_sentence_count"], "training_sentence_count", 1)
    tokens = integer(raw["training_token_count"], "training_token_count", 1)
    pseudo_sentences = integer(raw["pseudo_sentence_count"], "pseudo_sentence_count")
    pseudo_tokens = integer(raw["pseudo_token_count"], "pseudo_token_count")
    if (
        not 0 <= pseudo_sentences < sentences
        or pseudo_tokens > tokens
        or (pseudo_sentences == 0) != (pseudo_tokens == 0)
        or pseudo_tokens < pseudo_sentences
        or tokens - pseudo_tokens < sentences - pseudo_sentences
    ):
        raise ValidationError("Training/pseudo sentence and token counts disagree")
    reason = raw["termination_reason"]
    iterations = integer(raw["iterations"], "iterations")
    limit = config.training.parameters.max_iterations
    if (
        reason not in ("converged", "iteration_limit")
        or (limit > 0 and iterations > limit)
        or (reason == "iteration_limit" and (limit == 0 or iterations != limit))
    ):
        raise ValidationError("Invalid termination reason/iteration count")
    _equal_json(raw["python_version"], platform.python_version(), "python_version")
    _equal_json(raw["unicode_version"], unicodedata.unidata_version, "unicode_version")
    versions = dependency_versions()
    _equal_json(raw["dependency_versions"], dict(versions), "dependency_versions")
    training = TrainingRecord(
        sha256_value(raw["training_dataset_sha256"], "training_dataset_sha256"),
        sentences,
        tokens,
        pseudo_sentences,
        pseudo_tokens,
        tuple(sorted(expected.items())),
        config_sha256(config),
        config.target_inventory,
        seed,
        platform.python_version(),
        unicodedata.unidata_version,
        versions,
        reason,
        iterations,
    )
    return BundleMetadata(
        _identifier(raw["run_id"], "run_id"),
        _identifier(raw["candidate_id"], "candidate_id"),
        sha256_value(raw["model_sha256"], "model_sha256"),
        config,
        training,
    )


def _dispose(model: Any) -> None:
    if type(model) is CRF:
        if model._tagger is not None:
            model._tagger.close()
            model._tagger = None
        # Never delete a filename supplied by an invalid serialized estimator.
        if model.modelfile.auto and not model.modelfile.keep_tempfiles:
            model.modelfile.cleanup()


def _validate_model(model: Any, metadata: BundleMetadata) -> None:
    if type(model) is not CRF:
        raise ValidationError("Serialized object is not a CRF estimator")
    if not model.modelfile.auto or model.modelfile.keep_tempfiles:
        raise ValidationError("Model must contain its own temporary native bytes")
    _equal_json(
        model.get_params(),
        dict(metadata.training.crf_parameters),
        "estimator parameters",
    )
    embedded = getattr(model, "postag_training_record_", None)
    if type(embedded) is not TrainingRecord or embedded != metadata.training:
        raise ValidationError("Embedded training evidence disagrees with sidecar")
    stop = optimizer_termination(
        model.training_log_.log, metadata.config.training.parameters.max_iterations
    )
    if (stop.reason, stop.iterations) != (
        metadata.training.termination_reason,
        metadata.training.iterations,
    ):
        raise ValidationError("Optimizer log disagrees with sidecar")
    labels = model.classes_
    if (
        not labels
        or len(set(labels)) != len(labels)
        or not set(labels) <= set(metadata.config.target_inventory)
    ):
        raise ValidationError("Learned model labels disagree with target inventory")


def _signature(
    model: Any, config: PipelineConfig, monitor: MemoryMonitor
) -> list[object]:
    signatures: list[object] = []
    for forms in _PROBES:
        monitor.checkpoint()
        features = sentence_features(
            forms,
            schema_version=config.features.schema_version,
            normalizer_version=config.preprocessing.normalizer_version,
        )
        labels = model.predict_single(features)
        marginals = model.predict_marginals_single(features)
        if len(labels) != len(forms) or len(marginals) != len(forms):
            raise ValidationError("Unaligned round-trip prediction")
        for label, marginal in zip(labels, marginals, strict=True):
            if (
                label not in marginal
                or set(marginal) != set(model.classes_)
                or any(
                    type(p) not in (float, int)
                    or not math.isfinite(p)
                    or not 0 <= p <= 1
                    for p in marginal.values()
                )
                or not math.isclose(
                    sum(marginal.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6
                )
            ):
                raise ValidationError("Invalid round-trip marginal probabilities")
        signatures.append((labels, marginals))
    return signatures


def _deserialize(stream: BinaryIO, metadata: BundleMetadata) -> Any:
    model = None
    try:
        stream.seek(0)
        model = joblib.load(stream)
        _validate_model(model, metadata)
        return model
    except MemoryError, ResourceMonitoringError:
        _dispose(model)
        raise
    except Exception as exc:
        _dispose(model)
        raise ValidationError(f"Invalid serialized model: {exc}") from exc
    except BaseException:
        _dispose(model)
        raise


def _digest(
    stream: BinaryIO, buffer_bytes: int, monitor: MemoryMonitor | None = None
) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while chunk := stream.read(buffer_bytes):
        check_memory(monitor)
        digest.update(chunk)
    return digest.hexdigest()


def save_bundle(
    candidate: FittedCandidate,
    *,
    config: PipelineConfig,
    repository_root: Path,
    run_id: str,
    candidate_id: str,
    monitor: MemoryMonitor,
) -> BundleMetadata:
    """Publish a new verified pair after successful training and persistence monitors."""
    monitor.ensure_success()
    if monitor.budget_mib != config.resources.memory_budget_mib:
        raise ValidationError("Training monitor budget disagrees with configuration")
    metadata = BundleMetadata(
        _identifier(run_id, "run_id"),
        _identifier(candidate_id, "candidate_id"),
        "0" * 64,
        config,
        candidate.record,
    )
    # Encode/parse the exact snapshot before allocating anything.
    metadata = parse_metadata(loads_json(json_bytes(metadata.to_dict())))
    if candidate.record != metadata.training:
        raise ValidationError("Training evidence/configuration mismatch")
    _validate_model(candidate.model, metadata)
    root = repository_root.resolve()
    directory = root / "models" / metadata.run_id / "candidates" / metadata.candidate_id
    validate_output_path(directory, root)
    relative = directory.relative_to(root)
    with _parent_fd(relative, root, create=True) as parent:
        os.mkdir(relative.name, mode=0o700, dir_fd=parent)
    operation = MemoryMonitor(config.resources.memory_budget_mib)

    def validate_serialized(stream: BinaryIO) -> None:
        operation.ensure_success()
        if _digest(stream, config.resources.read_buffer_bytes) != metadata.model_sha256:
            raise ValidationError("Serialized bytes changed before publication")

    with atomic_writer(
        directory / "model.joblib", root, validate=validate_serialized
    ) as stream:
        with operation:
            before = _signature(candidate.model, config, operation)
            joblib.dump(candidate.model, stream, compress=0)
            stream.flush()
            operation.checkpoint()
            metadata = replace(
                metadata,
                model_sha256=_digest(
                    stream, config.resources.read_buffer_bytes, operation
                ),
            )
            restored = _deserialize(stream, metadata)
            try:
                if _signature(restored, config, operation) != before:
                    raise ValidationError(
                        "Round-trip predictions or confidences changed"
                    )
            finally:
                _dispose(restored)
        operation.ensure_success()
    content = json_bytes(metadata.to_dict())

    def validate_sidecar(stream: BinaryIO) -> None:
        operation.ensure_success()
        monitor.ensure_success()
        if parse_metadata(loads_json(stream.read())) != metadata:
            raise ValidationError("Metadata changed before publication")

    publish_bytes(directory / "metadata.json", content, root, validate=validate_sidecar)
    return metadata


@contextmanager
def _open_file(directory: Path, name: str) -> Iterator[BinaryIO]:
    absolute = directory.absolute()
    anchor = Path(absolute.anchor)
    reject_symlink_ancestors(absolute, anchor)
    relative = absolute.relative_to(anchor) / name
    with _parent_fd(relative, anchor, create=False) as parent:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValidationError(f"{name}: expected regular file")
            yield stream


def load_bundle(
    directory: Path,
    *,
    repository_root: Path | None = None,
    expected_run_id: str | None = None,
    expected_candidate_id: str | None = None,
    expected_model_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
    monitor: MemoryMonitor | None = None,
) -> LoadedBundle:
    """Load a trusted standalone bundle; optional frozen references bind its identity."""
    model = None
    try:
        with _open_file(directory, "metadata.json") as stream:
            content = stream.read()
        sidecar_sha = hashlib.sha256(content).hexdigest()
        if expected_metadata_sha256 is not None and sidecar_sha != sha256_value(
            expected_metadata_sha256, "expected_metadata_sha256"
        ):
            raise ValidationError("Metadata checksum mismatch")
        metadata = parse_metadata(loads_json(content))
        for actual, expected, field in (
            (metadata.run_id, expected_run_id, "run_id"),
            (metadata.candidate_id, expected_candidate_id, "candidate_id"),
            (metadata.model_sha256, expected_model_sha256, "model_sha256"),
        ):
            if expected is not None and actual != expected:
                raise ValidationError(f"Bundle {field} mismatch")
        if repository_root is not None:
            location = (
                repository_root.resolve()
                / "models"
                / metadata.run_id
                / "candidates"
                / metadata.candidate_id
            )
            if directory.absolute() != location:
                raise ValidationError(
                    "Bundle location disagrees with run/candidate identity"
                )
        active = (
            monitor
            if monitor is not None
            else MemoryMonitor(metadata.config.resources.memory_budget_mib)
        )

        def read_model() -> Any:
            check_memory(
                active, expected_budget_mib=metadata.config.resources.memory_budget_mib
            )
            with (
                _open_file(directory, "model.joblib") as original,
                tempfile.TemporaryFile("w+b") as snapshot,
            ):
                digest = hashlib.sha256()
                while chunk := original.read(
                    metadata.config.resources.read_buffer_bytes
                ):
                    active.checkpoint()
                    digest.update(chunk)
                    snapshot.write(chunk)
                if digest.hexdigest() != metadata.model_sha256:
                    raise ValidationError("Model checksum mismatch")
                result = _deserialize(snapshot, metadata)
                try:
                    active.checkpoint()
                except BaseException:
                    _dispose(result)
                    raise
                return result

        if monitor is None:
            with active:
                model = read_model()
            active.ensure_success()
        else:
            model = read_model()
        return LoadedBundle(model, metadata, sidecar_sha)
    except MemoryError, ResourceMonitoringError:
        _dispose(model)
        raise
    except Exception as exc:
        _dispose(model)
        raise ValidationError(f"{directory}: invalid model bundle: {exc}") from exc
    except BaseException:
        _dispose(model)
        raise
