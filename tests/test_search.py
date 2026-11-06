"""Offline baseline integration and specification-derived gate-ranking fixtures."""

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crf.data.identity import InputIdentity, sentence_id
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.evaluation import ModelReference
from crf.models import bundles, reuse, sampling, search
from crf.models.bundles import load_bundle
from crf.models.search import (
    BaselineTrial,
    SearchIncomplete,
    search_baseline,
    select_gate,
)
from crf.models.training import candidate_parameters
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryBudgetExceeded, MemoryMonitor
from crf.utils.serialization import jsonl_bytes, loads_json


def records(config: PipelineConfig, role: str) -> tuple[SentenceRecord, ...]:
    source = InputIdentity(
        1, role, getattr(config.data, role).as_posix(), "a" * 64, 100, "utf-8"
    )
    forms = ("Кіт", "спить") if role == "train_gold" else ("Пес", "біжить")
    result = []
    for ordinal in range(1, 7 if role == "train_gold" else 3):
        record = SentenceRecord(
            1,
            "gold",
            sentence_id(role, source.source_path, source.source_sha256, ordinal),
            source.source_path,
            source.source_sha256,
            role,
            ordinal,
            None,
            " ".join(forms),
            "original",
            (1, 2),
            forms,
            forms,
            (None, None),
            ("NOUN", "VERB"),
            ("NOUN", "VERB"),
            None,
        )
        result.append(
            parse_sentence_record(record.to_dict(), config=config, source=source)
        )
    return tuple(result)


@pytest.fixture(autouse=True)
def controlled_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (search, bundles, reuse, sampling):
        monkeypatch.setattr(
            module,
            "MemoryMonitor",
            lambda budget: MemoryMonitor(
                budget, sampler=lambda: 1, interval_seconds=60
            ),
        )


def execute(root: Path, config: PipelineConfig) -> Any:
    return search_baseline(
        records(config, "train_gold"),
        records(config, "val_gold"),
        config=config,
        repository_root=root,
        run_id="run",
    )


@pytest.mark.integration
def test_default_nine_fits_and_persisted_caches(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    models = []
    original = search.fit_candidate

    def spy(records_arg: Any, **kwargs: Any) -> Any:
        if models:
            assert models[-1]._tagger is None
            assert models[-1].modelfile.name is None
        materialized = tuple(records_arg)
        assert all(r.source_role == "train_gold" for r in materialized)
        assert all(r.feature_forms == ("Кіт", "спить") for r in materialized)
        calls.append((kwargs["c1"], kwargs["c2"]))
        fitted = original(materialized, **kwargs)
        models.append(fitted.model)
        return fitted

    monkeypatch.setattr(search, "fit_candidate", spy)
    result = execute(artifact_repository, artifact_config)
    assert calls == [(a, b) for a in (0.01, 0.1, 1.0) for b in (0.01, 0.1, 1.0)]
    assert len(result.trials) == 9
    assert all(t.status == "fitted" and t.error is None for t in result.trials)
    train_bytes = b"".join(
        jsonl_bytes(r.to_dict()) for r in records(artifact_config, "train_gold")
    )
    assert (
        artifact_repository / "data/processed/run/baseline_train.jsonl"
    ).read_bytes() == train_bytes
    assert {t.training_dataset_sha256 for t in result.trials} == {
        hashlib.sha256(train_bytes).hexdigest()
    }
    assert [
        loads_json(line) for line in result.trials_path.read_bytes().splitlines()
    ] == [t.to_dict() for t in result.trials]
    for trial in result.trials:
        ref = trial.model_ref
        assert ref is not None
        loaded = load_bundle(
            artifact_repository / "models/run/candidates" / ref.candidate_id,
            expected_model_sha256=ref.model_sha256,
            expected_metadata_sha256=ref.metadata_sha256,
        )
        try:
            assert loaded.metadata.training.pseudo_token_count == 0
            assert (
                loaded.metadata.training.training_dataset_sha256
                == trial.training_dataset_sha256
            )
        finally:
            loaded.close()
        metrics = loads_json(
            (
                artifact_repository
                / "reports/run/validation"
                / trial.trial_id
                / "metrics.json"
            ).read_bytes()
        )["models"][trial.trial_id]
        assert metrics["accuracy"] == trial.validation_accuracy
        assert metrics["macro_f1"] == trial.validation_macro_f1
        assert metrics["oov"]["token_count"] == 4
    assert result.gate == select_gate(result.trials, artifact_config)
    assert not (artifact_repository / "models/run/selected.json").exists()
    assert not (artifact_repository / "reports/run/run.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        execute(artifact_repository, artifact_config)
    assert len(calls) == 9


def trials(config: PipelineConfig) -> tuple[BaselineTrial, ...]:
    return tuple(
        BaselineTrial(
            "run",
            f"baseline_{i:03}",
            tuple(sorted(candidate_parameters(config, a, b).items())),
            "a" * 64,
            "fitted",
            ModelReference(f"baseline_{i:03}", "b" * 64, "c" * 64),
            0.8,
            0.4,
            None,
        )
        for i, (a, b) in enumerate(
            (a, b)
            for a in sorted(config.training.c1_grid)
            for b in sorted(config.training.c2_grid)
        )
    )


@pytest.mark.parametrize(
    "mode,index",
    [("regularization", 0), ("accuracy", 8), ("macro_f1", 7), ("unrounded", 6)],
)
def test_selection_tie_breakers(
    artifact_config: PipelineConfig, mode: str, index: int
) -> None:
    values = list(trials(artifact_config))
    if mode == "accuracy":
        values[index] = replace(
            values[index], validation_accuracy=0.9, validation_macro_f1=0.1
        )
    elif mode == "macro_f1":
        values[index] = replace(values[index], validation_macro_f1=0.5)
    elif mode == "unrounded":
        values[index] = replace(values[index], validation_accuracy=0.80000000000001)
    assert select_gate(tuple(values), artifact_config) == values[index].model_ref


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "duplicate",
        "failed",
        "not_executed",
        "dataset",
        "grid",
        "nan",
        "null",
        "bool",
    ],
)
def test_incomplete_or_inconsistent_search_cannot_select(
    artifact_config: PipelineConfig, mode: str
) -> None:
    values = list(trials(artifact_config))
    if mode == "missing":
        values.pop()
    elif mode == "duplicate":
        values[1] = values[0]
    elif mode in ("failed", "not_executed"):
        values[0] = replace(values[0], status=mode)
    elif mode == "dataset":
        values[0] = replace(values[0], training_dataset_sha256="d" * 64)
    elif mode == "grid":
        values[0] = replace(values[0], crf_parameters=values[1].crf_parameters)
    elif mode == "nan":
        values[0] = replace(values[0], validation_accuracy=float("nan"))
    elif mode == "null":
        values[0] = replace(values[0], model_ref=None)
    else:
        values[0] = replace(values[0], validation_accuracy=True)
    with pytest.raises(ValueError):
        select_gate(tuple(values), artifact_config)


@pytest.mark.integration
@pytest.mark.parametrize("stage", ["fit", "validation", "bundle", "metrics"])
def test_failure_retains_full_planned_ledger(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    name = dict(
        fit="fit_candidate",
        validation="score_validation",
        bundle="save_bundle",
        metrics="publish_validation_report",
    )[stage]
    original = getattr(search, name)
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryBudgetExceeded("controlled candidate failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(search, name, fail)
    with pytest.raises(SearchIncomplete) as exc:
        execute(artifact_repository, artifact_config)
    ledger = exc.value.trials
    assert [t.status for t in ledger] == ["fitted", "failed"] + ["not_executed"] * 7
    assert ledger[1].model_ref is None and ledger[1].validation_accuracy is None
    assert ledger[1].error[0] == stage
    assert (
        len(
            (artifact_repository / "reports/run/baseline/trials.jsonl")
            .read_bytes()
            .splitlines()
        )
        == 9
    )
    assert calls == 2


@pytest.mark.parametrize("mode", ["test", "empty", "duplicate", "late_error", "pseudo"])
def test_input_failures_prevent_all_fits(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    train = records(artifact_config, "train_gold")
    val: Any = records(artifact_config, "val_gold")
    if mode == "test":
        val = [replace(val[0], source_role="test_gold")]
    elif mode == "empty":
        val = []
    elif mode == "duplicate":
        val = [val[0], val[0]]
    elif mode == "pseudo":
        train = (replace(train[0], record_type="pseudo"),)
    else:
        original_val = val

        def late() -> Any:
            yield original_val[0]
            raise OSError("late decode failure")

        val = late()
    monkeypatch.setattr(
        search,
        "fit_candidate",
        lambda *args, **kwargs: pytest.fail("Invalid input reached fit"),
    )
    with pytest.raises(SearchIncomplete) as exc:
        search_baseline(
            train,
            val,
            config=artifact_config,
            repository_root=artifact_repository,
            run_id="run",
        )
    assert len(exc.value.trials) == 9 and all(
        t.status == "not_executed" for t in exc.value.trials
    )
    assert not (
        artifact_repository / "data/processed/run/baseline_train.jsonl"
    ).exists()


def test_preparation_resource_failure_prevents_dataset_publication(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search,
        "MemoryMonitor",
        lambda budget: MemoryMonitor(budget, sampler=lambda: budget * 1024**2 + 1),
    )
    with pytest.raises(SearchIncomplete) as exc:
        execute(artifact_repository, artifact_config)
    assert all(
        t.status == "not_executed" and t.training_dataset_sha256 is None
        for t in exc.value.trials
    )
    assert not (
        artifact_repository / "data/processed/run/baseline_train.jsonl"
    ).exists()


def test_ledger_failure_cannot_return_gate(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("ledger publication failed")

    monkeypatch.setattr(search, "publish_bytes", fail)
    with pytest.raises(OSError, match="ledger publication"):
        execute(
            artifact_repository,
            replace(
                artifact_config,
                training=replace(
                    artifact_config.training, c1_grid=(0.1,), c2_grid=(0.1,)
                ),
            ),
        )
    assert not (artifact_repository / "reports/run/baseline/trials.jsonl").exists()


@pytest.mark.integration
def test_mapped_custom_grid_is_frozen_and_sorted(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    config = replace(
        artifact_config,
        tag_mapping=tuple((tag, "FUNC") for tag in ("ADP", "CCONJ", "PART", "SCONJ")),
        training=replace(artifact_config.training, c1_grid=(1.0, 0.1), c2_grid=(0.1,)),
    )
    result = execute(artifact_repository, config)
    assert [dict(t.crf_parameters)["c1"] for t in result.trials] == [0.1, 1.0]
    assert len(result.trials) == 2
    assert all(t.validation_macro_f1 == pytest.approx(2 / 14) for t in result.trials)


@pytest.mark.integration
def test_empty_reuses_27_trials_without_fitting(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locked native CRF baseline only; reuse cannot fit or infer again."""
    baseline = execute(artifact_repository, artifact_config)
    datasets = sampling.freeze_datasets(
        records(artifact_config, "train_gold"),
        [],
        config=artifact_config,
        repository_root=artifact_repository,
        run_id="run",
        gate_model_sha256=baseline.gate.model_sha256,
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reuse must not fit or infer")

    monkeypatch.setattr(search, "fit_candidate", forbidden)
    monkeypatch.setattr(search, "score_validation", forbidden)
    result = reuse.reuse_empty_selections(
        datasets,
        baseline,
        config=artifact_config,
        repository_root=artifact_repository,
        run_id="run",
    )
    assert result.all_empty and result.retained_gate == baseline.gate
    assert len(result.trials) == 27
    assert len({t.trial_id for t in result.trials}) == 27
    for trial in result.trials:
        raw = trial.to_dict()
        assert raw["status"] == "reused" and raw["stage"] == "final"
        assert raw["model_ref"] == trial.baseline.model_ref.to_dict()
        assert raw["training_dataset_sha256"] == trial.baseline.training_dataset_sha256
        assert raw["validation_accuracy"] == trial.baseline.validation_accuracy
    report = loads_json(result.report_path.read_bytes())
    assert report["self_training_status"] == "not_performed_empty_selection"
    assert report["augmentation_gain_claimed"] is False


@pytest.mark.parametrize(
    "fault", ["dataset", "model", "metadata", "metrics", "missing", "grid", "gate"]
)
def test_reuse_rejects_bad_cache(
    artifact_repository: Path, artifact_config: PipelineConfig, fault: str
) -> None:
    config = replace(
        artifact_config,
        training=replace(artifact_config.training, c1_grid=(0.1,), c2_grid=(0.1,)),
    )
    baseline = execute(artifact_repository, config)
    datasets = sampling.freeze_datasets(
        records(config, "train_gold"),
        [],
        config=config,
        repository_root=artifact_repository,
        run_id="run",
        gate_model_sha256=baseline.gate.model_sha256,
    )
    directory = artifact_repository / "models/run/candidates/baseline_000"
    if fault == "dataset":
        datasets[0].path.write_bytes(b"{}\n")
    elif fault in {"model", "metadata"}:
        (
            directory / ("model.joblib" if fault == "model" else "metadata.json")
        ).write_bytes(b"corrupt")
    elif fault in {"metrics", "missing"}:
        path = artifact_repository / "reports/run/validation/baseline_000/metrics.json"
        if fault == "missing":
            path.unlink()
        else:
            path.write_bytes(b"{}")
    elif fault == "grid":
        baseline = replace(baseline, trials=())
    else:
        baseline = replace(baseline, gate=replace(baseline.gate, model_sha256="f" * 64))
    with pytest.raises((ValueError, OSError)):
        reuse.reuse_empty_selections(
            datasets,
            baseline,
            config=config,
            repository_root=artifact_repository,
            run_id="run",
        )
    assert not (artifact_repository / "reports/run/empty_reuse.json").exists()


def test_mixed_empty_reuses_only_empty_threshold(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    from crf.data.literature import RawSentence, iter_literature
    from crf.models.confidence import ConfidenceCandidate, SentenceConfidence

    config = replace(
        artifact_config,
        training=replace(artifact_config.training, c1_grid=(0.1,), c2_grid=(0.1,)),
    )
    baseline = execute(artifact_repository, config)
    raw = next(
        s
        for s in iter_literature(config, artifact_repository)
        if isinstance(s, RawSentence)
    )
    candidate = ConfidenceCandidate(
        raw,
        SentenceConfidence(
            ("NOUN",) * len(raw.feature_forms), (0.95,) * len(raw.feature_forms)
        ),
    )
    datasets = sampling.freeze_datasets(
        records(config, "train_gold"),
        [candidate],
        config=config,
        repository_root=artifact_repository,
        run_id="run",
        gate_model_sha256=baseline.gate.model_sha256,
    )
    result = reuse.reuse_empty_selections(
        datasets,
        baseline,
        config=config,
        repository_root=artifact_repository,
        run_id="run",
    )
    assert not result.all_empty and result.retained_gate is None
    assert [t.threshold for t in result.trials] == [0.98]
    report = loads_json(result.report_path.read_bytes())
    assert report["self_training_status"] == "nonempty_thresholds_pending"


def test_reuse_resource_and_existing_report(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crf.utils.paths import PathValidationError

    config = replace(
        artifact_config,
        training=replace(artifact_config.training, c1_grid=(0.1,), c2_grid=(0.1,)),
    )
    baseline = execute(artifact_repository, config)
    datasets = sampling.freeze_datasets(
        records(config, "train_gold"),
        [],
        config=config,
        repository_root=artifact_repository,
        run_id="run",
        gate_model_sha256=baseline.gate.model_sha256,
    )
    original = MemoryMonitor.__exit__

    def fail(self: MemoryMonitor, *args: Any) -> Any:
        original(self, *args)
        raise MemoryBudgetExceeded("reuse exit")

    with monkeypatch.context() as patch:
        patch.setattr(MemoryMonitor, "__exit__", fail)
        with pytest.raises(MemoryBudgetExceeded, match="reuse exit"):
            reuse.reuse_empty_selections(
                datasets,
                baseline,
                config=config,
                repository_root=artifact_repository,
                run_id="run",
            )
    path = artifact_repository / "reports/run/empty_reuse.json"
    assert not path.exists()
    first = reuse.reuse_empty_selections(
        datasets,
        baseline,
        config=config,
        repository_root=artifact_repository,
        run_id="run",
    )
    content = first.report_path.read_bytes()
    with pytest.raises((PathValidationError, FileExistsError)):
        reuse.reuse_empty_selections(
            datasets,
            baseline,
            config=config,
            repository_root=artifact_repository,
            run_id="run",
        )
    assert path.read_bytes() == content
