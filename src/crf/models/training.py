"""Sequential, monitored single-candidate CRF fitting without artifact publication."""

import hashlib
import logging
import math
import platform
import re
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import requires, version
from typing import Any, Literal

import pycrfsuite  # type: ignore[import-untyped]
from sklearn_crfsuite import CRF  # type: ignore[import-untyped]

from crf.data.normalization import normalize_forms
from crf.data.records import SentenceRecord
from crf.features.schema import FeatureDict
from crf.features.surface import sentence_features
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryMonitor, check_memory
from crf.utils.serialization import config_sha256, jsonl_bytes

logger = logging.getLogger(__name__)
_FIT_LOCK = threading.Lock()
type ParameterValue = str | int | float | bool | None


class TrainingError(ValueError):
    """Invalid training input or unsuccessful/unsupported optimization."""


@dataclass(frozen=True)
class Termination:
    reason: Literal["converged", "iteration_limit"]
    iterations: int


@dataclass(frozen=True)
class TrainingRecord:
    training_dataset_sha256: str
    training_sentence_count: int
    training_token_count: int
    pseudo_sentence_count: int
    pseudo_token_count: int
    crf_parameters: tuple[tuple[str, ParameterValue], ...]
    effective_config_sha256: str
    target_inventory: tuple[str, ...]
    seed: int
    python_version: str
    unicode_version: str
    dependency_versions: tuple[tuple[str, str], ...]
    termination_reason: Literal["converged", "iteration_limit"]
    iterations: int


@dataclass(frozen=True)
class FittedCandidate:
    # The estimator is necessarily mutable; the result binds it to immutable evidence.
    model: Any
    record: TrainingRecord


def optimizer_termination(messages: Iterable[str], max_iterations: int) -> Termination:
    """Fail closed on optimizer errors, invalid losses and ambiguous stop evidence."""
    iterations = 0
    losses = 0
    reasons: list[Literal["converged", "iteration_limit"]] = []
    for message in messages:
        for line in message.splitlines():
            if "terminated with error" in line.lower():
                raise TrainingError(f"Optimizer failed: {line}")
            if line.startswith("***** Iteration #"):
                match = re.fullmatch(r"\*{5} Iteration #(\d+) \*{5}", line)
                if match is None or int(match[1]) != iterations + 1:
                    raise TrainingError("Invalid optimizer iteration sequence")
                if losses != iterations:
                    raise TrainingError("Missing optimizer objective")
                iterations += 1
            if line.startswith("Loss:"):
                try:
                    loss = float(line.removeprefix("Loss:").strip())
                except ValueError as exc:
                    raise TrainingError("Invalid optimizer objective") from exc
                if not math.isfinite(loss) or losses != iterations - 1:
                    raise TrainingError("Nonfinite or misplaced optimizer objective")
                losses += 1
            if line == "L-BFGS terminated with the maximum number of iterations":
                reasons.append("iteration_limit")
            elif line in (
                "L-BFGS resulted in convergence",
                "L-BFGS terminated with the stopping criteria",
            ):
                reasons.append("converged")
    if len(reasons) != 1 or losses != iterations:
        raise TrainingError("Missing or ambiguous optimizer termination evidence")
    reason = reasons[0]
    if (max_iterations > 0 and iterations > max_iterations) or (
        reason == "iteration_limit"
        and (max_iterations == 0 or iterations != max_iterations)
    ):
        raise TrainingError("Optimizer iterations disagree with configured limit")
    return Termination(reason, iterations)


class _MonitoredTrainer(pycrfsuite.Trainer):  # type: ignore[misc]
    def __init__(self, *, monitor: MemoryMonitor, **kwargs: Any) -> None:
        self.monitor = monitor
        super().__init__(**kwargs)

    def append(self, xseq: object, yseq: object, group: int = 0) -> None:
        self.monitor.checkpoint()
        super().append(xseq, yseq, group)

    def message(self, message: str) -> None:
        self.monitor.checkpoint()
        # Trainer's parser records logs even with verbose=False; stdout stays clean.
        super().message(message)


def _parameters(
    config: PipelineConfig, c1: float, c2: float
) -> dict[str, ParameterValue]:
    for name, value, grid in (
        ("c1", c1, config.training.c1_grid),
        ("c2", c2, config.training.c2_grid),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or value not in grid
        ):
            raise TrainingError(f"{name}: expected a finite frozen-grid value")
    return {
        "algorithm": "lbfgs",
        **asdict(config.training.parameters),
        "c1": float(c1),
        "c2": float(c2),
        "linesearch": "MoreThuente",
        "max_linesearch": 20,
        "verbose": False,
        "model_filename": None,
        "keep_tempfiles": False,
        "trainer_cls": None,
        **dict.fromkeys(
            (
                "calibration_eta",
                "calibration_rate",
                "calibration_samples",
                "calibration_candidates",
                "calibration_max_trials",
                "pa_type",
                "c",
                "error_sensitive",
                "averaging",
                "variance",
                "gamma",
            )
        ),
    }


def _validate_record(
    record: SentenceRecord, config: PipelineConfig, allow_pseudo: bool
) -> None:
    if not isinstance(record, SentenceRecord):
        raise TrainingError("Training requires validated SentenceRecord objects")
    if record.record_type == "gold" and record.source_role == "train_gold":
        if (
            record.source_path != config.data.train_gold.as_posix()
            or record.pseudo is not None
        ):
            raise TrainingError(
                "Supervised record disagrees with training configuration"
            )
        if (
            record.original_upos is None
            or config.label_space.map_upos(record.original_upos) != record.target_labels
        ):
            raise TrainingError("Supervised targets disagree with configured mapping")
    elif (
        allow_pseudo
        and record.record_type == "pseudo"
        and record.source_role == "literature"
    ):
        if record.pseudo is None or record.original_upos is not None:
            raise TrainingError("Invalid pseudo-label training record")
    else:
        raise TrainingError(
            "Only declared train_gold and explicitly enabled pseudo records may be fitted"
        )
    count = len(record.token_ids)
    if not count or any(
        len(values) != count
        for values in (
            record.original_forms,
            record.normalized_forms,
            record.target_labels,
        )
    ):
        raise TrainingError("Training sentence must be nonempty and aligned")
    config.label_space.validate_targets(record.target_labels)
    if (
        normalize_forms(
            record.original_forms, version=config.preprocessing.normalizer_version
        )
        != record.feature_forms
    ):
        raise TrainingError("Training forms disagree with normalization")


def _dependency_versions() -> tuple[tuple[str, str], ...]:
    # Include direct runtime dependencies plus the native optimizer and its adapter.
    names = {"ukrainian-crf-postag", "python-crfsuite", "scipy", "scikit-learn"}
    for requirement in requires("ukrainian-crf-postag") or ():
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match:
            names.add(match[0])
    return tuple((name, version(name)) for name in sorted(names))


def _fit(
    records: Iterable[SentenceRecord],
    config: PipelineConfig,
    parameters: dict[str, ParameterValue],
    allow_pseudo: bool,
    monitor: MemoryMonitor,
) -> FittedCandidate:
    check_memory(monitor, expected_budget_mib=config.resources.memory_budget_mib)
    features: list[list[FeatureDict]] = []
    targets: list[list[str]] = []
    digest = hashlib.sha256()
    token_count = pseudo_sentences = pseudo_tokens = 0
    for record in records:
        monitor.checkpoint()
        _validate_record(record, config, allow_pseudo)
        digest.update(jsonl_bytes(record.to_dict()))
        features.append(
            sentence_features(
                record.feature_forms,
                schema_version=config.features.schema_version,
                normalizer_version=config.preprocessing.normalizer_version,
            )
        )
        targets.append(list(record.target_labels))
        token_count += len(record.target_labels)
        if record.record_type == "pseudo":
            pseudo_sentences += 1
            pseudo_tokens += len(record.target_labels)
    if not features or pseudo_sentences == len(features):
        raise TrainingError("Training requires at least one supervised sentence")
    monitor.checkpoint()
    model = CRF(**parameters)
    model.trainer_cls = partial(_MonitoredTrainer, monitor=monitor)
    try:
        logger.info(
            "Fitting CRF on %d sentences / %d tokens", len(features), token_count
        )
        model.fit(features, targets)
        monitor.checkpoint()
        termination = optimizer_termination(
            model.training_log_.log, config.training.parameters.max_iterations
        )
        if model.tagger_ is None:
            raise TrainingError("Optimizer produced no usable model")
        evidence = TrainingRecord(
            digest.hexdigest(),
            len(features),
            token_count,
            pseudo_sentences,
            pseudo_tokens,
            tuple(sorted(parameters.items())),
            config_sha256(config),
            config.target_inventory,
            config.seed,
            platform.python_version(),
            unicodedata.unidata_version,
            _dependency_versions(),
            termination.reason,
            termination.iterations,
        )
        return FittedCandidate(model, evidence)
    except BaseException:
        if model._tagger is not None:
            model._tagger.close()
        model.modelfile.cleanup()
        raise
    finally:
        # Never retain a monitor/lock or a local callable in a serializable estimator.
        model.trainer_cls = None


def fit_candidate(
    records: Iterable[SentenceRecord],
    *,
    config: PipelineConfig,
    c1: float,
    c2: float,
    allow_pseudo: bool = False,
    monitor: MemoryMonitor | None = None,
) -> FittedCandidate:
    """Fit one fresh candidate; shared-monitor results require later successful exit.

    Input producers must supply validated records. No source files are opened and
    no artifacts are published. Native fitting holds its data in memory.
    """
    if threading.current_thread() is not threading.main_thread():
        raise TrainingError(
            "Candidate fitting must run sequentially on the main thread"
        )
    if type(allow_pseudo) is not bool:
        raise TrainingError("allow_pseudo must be boolean")
    parameters = _parameters(config, c1, c2)
    if not _FIT_LOCK.acquire(blocking=False):
        raise TrainingError("Another candidate fit is already active")
    result: FittedCandidate | None = None
    try:
        if monitor is not None:
            return _fit(records, config, parameters, allow_pseudo, monitor)
        owned_monitor = MemoryMonitor(config.resources.memory_budget_mib)
        with owned_monitor:
            result = _fit(records, config, parameters, allow_pseudo, owned_monitor)
        owned_monitor.ensure_success()
        return result
    except BaseException:
        if result is not None:
            if result.model._tagger is not None:
                result.model._tagger.close()
            result.model.modelfile.cleanup()
        raise
    finally:
        _FIT_LOCK.release()
