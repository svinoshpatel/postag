"""Offline training contract fixtures; native integration uses locked CRFsuite only."""

import hashlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crf.data.identity import InputIdentity, sentence_id
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.features.surface import sentence_features
from crf.models import training
from crf.models.training import TrainingError, fit_candidate, optimizer_termination
from crf.utils.config import PipelineConfig
from crf.utils.resources import (
    MemoryBudgetExceeded,
    MemoryMonitor,
    ResourceMonitoringError,
)
from crf.utils.serialization import config_sha256, jsonl_bytes


def record(
    config: PipelineConfig, ordinal: int = 1, *, pseudo: bool = False
) -> SentenceRecord:
    source = InputIdentity(
        1,
        "literature" if pseudo else "train_gold",
        "data/raw/literature/example.txt"
        if pseudo
        else config.data.train_gold.as_posix(),
        "a" * 64,
        100,
        "utf-8",
    )
    labels = config.label_space.map_upos(("NOUN", "VERB"))
    value = SentenceRecord(
        1,
        "gold",
        sentence_id(
            source.source_role, source.source_path, source.source_sha256, ordinal
        ),
        source.source_path,
        source.source_sha256,
        source.source_role,
        ordinal,
        None,
        "Кіт спить",
        "original",
        (1, 2),
        ("Кіт", "спить"),
        ("Кіт", "спить"),
        (None, None),
        ("NOUN", "VERB"),
        labels,
        None,
    ).to_dict()
    if pseudo:
        from crf.data.identity import sampling_priority

        value.update(
            record_type="pseudo",
            paragraph_ordinal=1,
            original_upos=None,
            pseudo={
                "token_confidences": [0.95, 0.99],
                "min_confidence": 0.95,
                "sentence_fingerprint": "b" * 64,
                "sampling_priority": sampling_priority(config.seed, "b" * 64),
                "gate_model_sha256": "c" * 64,
            },
        )
    return parse_sentence_record(value, config=config, source=source)


@pytest.fixture
def monitor() -> Any:
    with MemoryMonitor(8192, interval_seconds=60, sampler=lambda: 1) as active:
        yield active
    active.ensure_success()


def fit(
    config: PipelineConfig, monitor: MemoryMonitor, records: Any = None, **kwargs: Any
) -> Any:
    return fit_candidate(
        [record(config)] if records is None else records,
        config=config,
        c1=0.1,
        c2=0.1,
        monitor=monitor,
        **kwargs,
    )


@pytest.mark.integration
def test_native_fit_settings_identity_and_fresh_model(
    artifact_config: PipelineConfig,
    monitor: MemoryMonitor,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = [record(artifact_config), record(artifact_config, 2)]
    result = fit(artifact_config, monitor, iter(records))
    evidence = result.record
    assert (
        evidence.training_dataset_sha256
        == hashlib.sha256(
            b"".join(jsonl_bytes(item.to_dict()) for item in records)
        ).hexdigest()
    )
    assert (evidence.training_sentence_count, evidence.training_token_count) == (2, 4)
    assert (evidence.pseudo_sentence_count, evidence.pseudo_token_count) == (0, 0)
    assert evidence.effective_config_sha256 == config_sha256(artifact_config)
    assert evidence.target_inventory == artifact_config.target_inventory
    assert evidence.seed == 42
    assert evidence.python_version and evidence.unicode_version
    assert dict(evidence.dependency_versions)["python-crfsuite"]
    params = dict(evidence.crf_parameters)
    assert params == result.model.get_params()
    assert {
        key: params[key]
        for key in (
            "algorithm",
            "c1",
            "c2",
            "max_iterations",
            "num_memories",
            "epsilon",
            "period",
            "delta",
            "min_freq",
            "all_possible_states",
            "all_possible_transitions",
        )
    } == dict(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        num_memories=6,
        epsilon=1e-5,
        period=10,
        delta=1e-5,
        min_freq=0.0,
        all_possible_states=False,
        all_possible_transitions=True,
    )
    assert "random_state" not in params
    assert result.model.trainer_cls is None
    assert evidence.termination_reason == "converged"
    assert 0 < evidence.iterations <= 200
    assert result.model.predict_single(sentence_features(["Кіт", "спить"])) == [
        "NOUN",
        "VERB",
    ]
    another = fit(artifact_config, monitor, records)
    assert another.model is not result.model
    assert another.model.modelfile.name != result.model.modelfile.name
    assert another.record == evidence
    assert capsys.readouterr().out == ""


@pytest.mark.integration
def test_native_iteration_limit(
    artifact_config: PipelineConfig, monitor: MemoryMonitor
) -> None:
    config = replace(
        artifact_config,
        training=replace(
            artifact_config.training,
            parameters=replace(artifact_config.training.parameters, max_iterations=1),
        ),
    )
    result = fit(config, monitor)
    assert result.record.termination_reason == "iteration_limit"
    assert result.record.iterations == 1


@pytest.mark.integration
def test_declared_augmented_input_only(
    artifact_config: PipelineConfig,
    monitor: MemoryMonitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [record(artifact_config), record(artifact_config, 2, pseudo=True)]
    calls = []
    original = training.CRF.fit

    def spy(self: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((x, y, args, kwargs))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(training.CRF, "fit", spy)
    result = fit(artifact_config, monitor, records, allow_pseudo=True)
    assert len(calls) == 1
    x, y, args, kwargs = calls[0]
    assert x == [sentence_features(item.feature_forms) for item in records]
    assert y == [["NOUN", "VERB"]] * 2
    assert args == () and kwargs == {}
    assert result.record.pseudo_sentence_count == 1
    assert result.record.pseudo_token_count == 2
    assert result.record.training_token_count == 4


@pytest.mark.parametrize(
    "mode",
    [
        "val_gold",
        "test_gold",
        "pseudo",
        "empty",
        "only_pseudo",
        "unaligned",
        "bad_mapping",
        "bad_path",
        "bad_forms",
    ],
)
def test_invalid_inputs_never_fit(
    artifact_config: PipelineConfig,
    monitor: MemoryMonitor,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    item = record(artifact_config)
    if mode in ("val_gold", "test_gold"):
        item = replace(item, source_role=mode)
    elif mode in ("pseudo", "only_pseudo"):
        item = record(artifact_config, pseudo=True)
    elif mode == "unaligned":
        item = replace(item, target_labels=("NOUN",))
    elif mode == "bad_mapping":
        item = replace(item, target_labels=("VERB", "NOUN"))
    elif mode == "bad_path":
        item = replace(item, source_path=artifact_config.data.val_gold.as_posix())
    elif mode == "bad_forms":
        item = replace(item, normalized_forms=("пес", "спить"))

    def forbidden(**kwargs: Any) -> None:
        pytest.fail("Invalid dataset reached estimator construction")

    monkeypatch.setattr(training, "CRF", forbidden)
    with pytest.raises(ValueError):
        fit(
            artifact_config,
            monitor,
            [] if mode == "empty" else [item],
            allow_pseudo=mode == "only_pseudo",
        )


def test_late_input_failure_never_fits(
    artifact_config: PipelineConfig,
    monitor: MemoryMonitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def records() -> Any:
        yield record(artifact_config)
        raise OSError("late source failure")

    monkeypatch.setattr(
        training,
        "CRF",
        lambda **kwargs: pytest.fail("Fit started before iterator exhausted"),
    )
    with pytest.raises(OSError, match="late source"):
        fit(artifact_config, monitor, records())
    # A failed call must release the sequential-fit lock.
    assert training._FIT_LOCK.acquire(blocking=False)
    training._FIT_LOCK.release()


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 0.2])
def test_invalid_candidate_grid_value(
    artifact_config: PipelineConfig, value: float
) -> None:
    with pytest.raises(TrainingError, match="frozen-grid"):
        fit_candidate([], config=artifact_config, c1=value, c2=0.1)


@pytest.mark.parametrize(
    "stop,reason",
    [
        ("L-BFGS terminated with the maximum number of iterations", "iteration_limit"),
        ("L-BFGS terminated with the stopping criteria", "converged"),
        ("L-BFGS resulted in convergence", "converged"),
    ],
)
def test_explicit_optimizer_stop(stop: str, reason: str) -> None:
    result = optimizer_termination(["***** Iteration #1 *****\nLoss: 1.25\n", stop], 1)
    assert result.reason == reason and result.iterations == 1


@pytest.mark.parametrize(
    "messages,limit",
    [
        ([], 200),
        (["L-BFGS terminated with error code (2)"], 200),
        (["L-BFGS stopped for unknown reason"], 200),
        (["***** Iteration #2 *****\nLoss: 1.0"], 200),
        (["***** Iteration #1 *****\nLoss: nan"], 200),
        (["***** Iteration #1 *****\nLoss: inf"], 200),
        (["***** Iteration #1 *****\nLoss: 1e999"], 200),
        (["***** Iteration #1 *****\nLoss: broken"], 200),
        (["***** Iteration #1 *****\nL-BFGS resulted in convergence"], 200),
        (
            [
                "***** Iteration #1 *****\nLoss: 1\nLoss: 1\nL-BFGS resulted in convergence"
            ],
            200,
        ),
        (
            [
                "***** Iteration #1 *****\nLoss: 1\nL-BFGS terminated with the maximum number of iterations"
            ],
            200,
        ),
        (
            [
                "***** Iteration #1 *****\nLoss: 1\nL-BFGS terminated with the maximum number of iterations"
            ],
            0,
        ),
        (["L-BFGS resulted in convergence\nL-BFGS resulted in convergence"], 200),
    ],
)
def test_invalid_optimizer_evidence(messages: list[str], limit: int) -> None:
    with pytest.raises(TrainingError):
        optimizer_termination(messages, limit)


def test_unlimited_optimizer_setting() -> None:
    assert (
        optimizer_termination(
            ["***** Iteration #1 *****\nLoss: 1\nL-BFGS resulted in convergence"], 0
        ).reason
        == "converged"
    )


@pytest.mark.parametrize("state", ["inactive", "finished", "wrong_budget"])
def test_reject_invalid_monitor(artifact_config: PipelineConfig, state: str) -> None:
    monitor = MemoryMonitor(1 if state == "wrong_budget" else 8192, sampler=lambda: 1)
    if state == "finished":
        with monitor:
            pass
    with pytest.raises(ResourceMonitoringError):
        fit(artifact_config, monitor)


@pytest.mark.integration
def test_optimizer_error_file_is_cleaned(
    artifact_config: PipelineConfig,
    monitor: MemoryMonitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    original = training.CRF.fit

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        paths.append(Path(self.modelfile.name))
        return result

    monkeypatch.setattr(training.CRF, "fit", spy)
    item = record(artifact_config)
    item = replace(item, target_labels=("NOUN", "NOUN"), original_upos=("NOUN", "NOUN"))
    with pytest.raises(TrainingError, match="Optimizer failed"):
        fit(artifact_config, monitor, [item])
    assert paths and not paths[0].exists()


def test_parallel_and_reentrant_calls_rejected(artifact_config: PipelineConfig) -> None:
    errors = []

    def worker() -> None:
        try:
            fit_candidate([], config=artifact_config, c1=0.1, c2=0.1)
        except TrainingError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "main thread" in errors[0]
    with training._FIT_LOCK, pytest.raises(TrainingError, match="already active"):
        fit_candidate([], config=artifact_config, c1=0.1, c2=0.1)


@pytest.mark.integration
def test_owned_monitor_exit_failure_discards_model(
    artifact_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExitFailure(MemoryMonitor):
        def __exit__(self, *args: Any) -> Any:
            super().__exit__(*args)
            raise MemoryBudgetExceeded("exit budget failure")

    monkeypatch.setattr(
        training, "MemoryMonitor", lambda budget: ExitFailure(budget, sampler=lambda: 1)
    )
    models = []
    original = training.CRF.fit

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        models.append((self, Path(self.modelfile.name)))
        return result

    monkeypatch.setattr(training.CRF, "fit", spy)
    with pytest.raises(MemoryBudgetExceeded, match="exit budget"):
        fit_candidate([record(artifact_config)], config=artifact_config, c1=0.1, c2=0.1)
    assert models and not models[0][1].exists()
    assert models[0][0].trainer_cls is None


@pytest.mark.integration
def test_custom_mapping_and_ordered_dataset_identity(
    artifact_config: PipelineConfig,
    monitor: MemoryMonitor,
) -> None:
    config = replace(
        artifact_config,
        tag_mapping=tuple((tag, "FUNC") for tag in ("ADP", "CCONJ", "PART", "SCONJ")),
    )
    first = record(config)
    value = first.to_dict()
    value.update(original_upos=["ADP", "NOUN"], target_labels=["FUNC", "NOUN"])
    source = InputIdentity(
        1, "train_gold", first.source_path, first.source_sha256, 100, "utf-8"
    )
    first = parse_sentence_record(value, config=config, source=source)
    second = record(config, 2)
    forward = fit(config, monitor, [first, second])
    reverse = fit(config, monitor, [second, first])
    assert (
        forward.record.training_dataset_sha256 != reverse.record.training_dataset_sha256
    )
    assert "FUNC" in forward.model.classes_ and "ADP" not in forward.model.classes_
    assert len(forward.record.target_inventory) == 14


@pytest.mark.integration
@pytest.mark.parametrize("stage", ["preparation", "append", "optimizer", "after_fit"])
def test_resource_failure_at_training_boundaries(
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    exceeded = False
    paths = []

    def sample() -> int:
        return 8192 * 1024**2 + 1 if exceeded else 1

    def trigger() -> None:
        nonlocal exceeded
        exceeded = True

    if stage == "preparation":
        original_features = training.sentence_features

        def features(*args: Any, **kwargs: Any) -> Any:
            result = original_features(*args, **kwargs)
            trigger()
            return result

        monkeypatch.setattr(training, "sentence_features", features)
    elif stage in ("append", "optimizer"):
        method = "append" if stage == "append" else "message"
        original_method = getattr(training._MonitoredTrainer, method)

        def callback(self: Any, *args: Any, **kwargs: Any) -> Any:
            trigger()
            return original_method(self, *args, **kwargs)

        monkeypatch.setattr(training._MonitoredTrainer, method, callback)
    original_fit = training.CRF.fit

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = original_fit(self, *args, **kwargs)
            if stage == "after_fit":
                trigger()
            return result
        finally:
            if self.modelfile.name:
                paths.append(Path(self.modelfile.name))

    monkeypatch.setattr(training.CRF, "fit", spy)
    with (
        pytest.raises(MemoryBudgetExceeded),
        MemoryMonitor(8192, sampler=sample, interval_seconds=60) as active,
    ):
        fit(artifact_config, active)
    assert all(not path.exists() for path in paths)


def test_owned_monitor_success(
    artifact_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = MemoryMonitor(8192, sampler=lambda: 1, interval_seconds=60)
    monkeypatch.setattr(training, "MemoryMonitor", lambda budget: active)
    result = fit_candidate(
        [record(artifact_config)], config=artifact_config, c1=0.1, c2=0.1
    )
    active.ensure_success()
    assert result.record.training_sentence_count == 1
