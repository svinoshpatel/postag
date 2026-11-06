"""Offline develop integration uses tiny CoNLL-U fixtures and locked native CRF."""

import importlib.util
import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from crf import experiments, pipeline
from crf.models import bundles, reuse, sampling, search
from crf.models.confidence import ConfidenceCandidate, SentenceConfidence
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryBudgetExceeded, MemoryMonitor
from crf.utils.serialization import canonical_config_bytes


@pytest.fixture(autouse=True)
def controlled_monitor(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    log = logging.getLogger("crf")
    previous = (log.handlers[:], log.level, log.propagate)
    for module in (pipeline, bundles, reuse, sampling, search, experiments):
        monkeypatch.setattr(
            module,
            "MemoryMonitor",
            lambda budget: MemoryMonitor(
                budget, sampler=lambda: 1, interval_seconds=60
            ),
        )

    yield
    log.handlers, level, log.propagate = previous
    log.setLevel(level)


@pytest.fixture
def development_repository(artifact_repository: Path) -> Path:
    for name, words in [
        ("train", ("Кіт", "спить")),
        ("val", ("Пес", "біжить")),
        ("test", ("Тест", "стоїть")),
    ]:
        # Test target fields intentionally invalid: development may audit text only.
        labels = ("NOUN", "VERB") if name != "test" else ("PRIVATE", "PRIVATE")
        content = ""
        for ordinal in range(1, 5):
            content += f"# sent_id = {name}-{ordinal}\n# text = {' '.join(words)}\n"
            for i, (form, label) in enumerate(zip(words, labels, strict=True), 1):
                content += f"{i}\t{form}\t_\t{label}\t_\t_\t0\troot\t_\t_\n"
            content += "\n"
        (artifact_repository / f"data/raw/gold/{name}.conllu").write_text(
            content, encoding="utf-8"
        )
    return artifact_repository


def config_for(config: PipelineConfig, *, full: bool = False) -> PipelineConfig:
    training = replace(
        config.training,
        parameters=replace(config.training.parameters, max_iterations=3),
    )
    if not full:
        training = replace(training, c1_grid=(0.1,), c2_grid=(0.1,))
    return replace(config, training=training)


def confidence(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    original = pipeline.iter_scored_candidates

    def forced(*args: Any, **kwargs: Any) -> Any:
        for candidate in original(*args, **kwargs):
            yield ConfidenceCandidate(
                candidate.sentence,
                SentenceConfidence(
                    candidate.confidence.labels,
                    (value,) * len(candidate.confidence.labels),
                ),
            )

    monkeypatch.setattr(pipeline, "iter_scored_candidates", forced)


def ledger(root: Path, run_id: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "reports" / run_id / "trials.jsonl")
        .read_bytes()
        .splitlines()
    ]


@pytest.mark.integration
def test_default_36_fresh_fits(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_for(artifact_config, full=True)
    confidence(monkeypatch, 1.0)
    models = []
    datasets: list[tuple[str, ...]] = []
    original = pipeline.fit_candidate

    def spy(records: Any, **kwargs: Any) -> Any:
        if models:
            assert models[-1]._tagger is None
            assert models[-1].modelfile.name is None
        materialized = tuple(records)
        assert all(r.source_role in {"train_gold", "literature"} for r in materialized)
        datasets.append(tuple(r.sentence_id for r in materialized))
        result = original(materialized, **kwargs)
        models.append(result.model)
        return result

    monkeypatch.setattr(pipeline, "fit_candidate", spy)
    monkeypatch.setattr(search, "fit_candidate", spy)
    # Observe labeled loader isolation explicitly as well as invalid test UPOS.
    original_loader = pipeline.iter_gold_records
    roles = []

    def loader(source: Any, *args: Any, **kwargs: Any) -> Any:
        roles.append(source.source_role)
        assert source.source_role != "test_gold"
        yield from original_loader(source, *args, **kwargs)

    monkeypatch.setattr(pipeline, "iter_gold_records", loader)
    result = pipeline.search_development(config, development_repository)
    raw = ledger(development_repository, result.run_id)
    assert roles == ["train_gold", "val_gold"]
    assert len(models) == len(raw) == 36
    assert len({id(m) for m in models}) == 36
    assert len({r["trial_id"] for r in raw}) == 36
    assert all(r["status"] == "fitted" for r in raw)
    assert [r["stage"] for r in raw].count("baseline") == 9
    assert [r["stage"] for r in raw].count("final") == 27
    for start in (9, 18, 27):
        assert all(d == datasets[start] for d in datasets[start : start + 9])
    assert all(m._tagger is None and m.modelfile.name is None for m in models)
    report = development_repository / "reports" / result.run_id
    assert (report / "effective_config.json").read_bytes() == canonical_config_bytes(
        config
    )
    assert not (report / "run.json").exists()
    assert not (
        development_repository / "models" / result.run_id / "selected.json"
    ).exists()


@pytest.mark.integration
def test_default_empty_reuse_only_nine_fits(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_for(artifact_config, full=True)
    confidence(monkeypatch, 0.1)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("empty final selection must not fit")

    monkeypatch.setattr(pipeline, "fit_candidate", forbidden)
    result = pipeline.search_development(config, development_repository)
    raw = ledger(development_repository, result.run_id)
    assert len(raw) == 36
    assert sum(r["status"] == "reused" for r in raw) == 27
    assert len({r["model_ref"]["candidate_id"] for r in raw}) == 9
    assert all(t.reused for t in result.final_trials)


@pytest.mark.parametrize(
    "point", ["fit", "bundle", "validation", "preparation", "baseline", "pseudo"]
)
def test_failure_keeps_full_plan(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
) -> None:
    config = config_for(artifact_config)
    confidence(monkeypatch, 1.0)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"controlled {point}")

    if point == "fit":
        monkeypatch.setattr(pipeline, "fit_candidate", fail)
    elif point == "bundle":
        monkeypatch.setattr(pipeline, "save_bundle", fail)
    elif point == "validation":
        monkeypatch.setattr(pipeline, "score_validation", fail)
    elif point == "preparation":
        monkeypatch.setattr(pipeline, "iter_gold_records", fail)
    elif point == "baseline":
        monkeypatch.setattr(search, "fit_candidate", fail)
    else:
        monkeypatch.setattr(pipeline, "iter_scored_candidates", fail)
    with pytest.raises(pipeline.DevelopmentIncomplete) as caught:
        pipeline.search_development(config, development_repository)
    raw = ledger(development_repository, caught.value.run_id)
    assert len(raw) == 4
    if point in {"fit", "bundle", "validation"}:
        assert [r["status"] for r in raw] == [
            "fitted",
            "failed",
            "not_executed",
            "not_executed",
        ]
        assert all(r["training_dataset_sha256"] is not None for r in raw)
    elif point == "baseline":
        assert [r["status"] for r in raw] == [
            "failed",
            "not_executed",
            "not_executed",
            "not_executed",
        ]
    elif point == "preparation":
        assert all(r["status"] == "not_executed" for r in raw)
    else:
        assert [r["status"] for r in raw] == [
            "fitted",
            "not_executed",
            "not_executed",
            "not_executed",
        ]
    assert not (
        development_repository / "reports" / caught.value.run_id / "run.json"
    ).exists()


def test_baseline_ledger_failure_preserves_actual_fit(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("baseline ledger unavailable")

    monkeypatch.setattr(search, "publish_bytes", fail)
    with pytest.raises(pipeline.DevelopmentIncomplete) as caught:
        pipeline.search_development(config_for(artifact_config), development_repository)
    raw = ledger(development_repository, caught.value.run_id)
    assert raw[0]["status"] == "fitted"
    assert all(r["status"] == "not_executed" for r in raw[1:])


def test_input_change_refuses_search_success(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confidence(monkeypatch, 0.1)
    original = pipeline.reuse_empty_selections

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        (development_repository / "data/raw/literature/а.txt").write_text(
            "Зміна", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(pipeline, "reuse_empty_selections", mutate)
    with pytest.raises(
        pipeline.DevelopmentIncomplete, match="Inputs changed"
    ) as caught:
        pipeline.search_development(config_for(artifact_config), development_repository)
    assert len(ledger(development_repository, caught.value.run_id)) == 4


def test_cli_seed_and_json(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pipeline_cli", root / "scripts/run_pipeline.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = config_for(artifact_config)
    path = development_repository / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.chdir(development_repository)
    confidence(monkeypatch, 0.1)
    assert module.main(["develop", "--config", str(path), "--seed", "73"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "complete"
    effective = json.loads(
        (
            development_repository
            / "reports"
            / output["run_id"]
            / "effective_config.json"
        ).read_bytes()
    )
    assert effective["seed"] == 73
    assert module.main(["develop", "--config", "missing.yaml"]) == 1
    assert capsys.readouterr().out == ""


def test_mixed_reuse_and_fresh_fits(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confidence(monkeypatch, 0.95)
    config = config_for(artifact_config)
    models = []
    original = pipeline.fit_candidate

    def spy(*args: Any, **kwargs: Any) -> Any:
        candidate = original(*args, **kwargs)
        models.append(candidate.model)
        return candidate

    monkeypatch.setattr(pipeline, "fit_candidate", spy)
    result = pipeline.search_development(config, development_repository)
    raw = ledger(development_repository, result.run_id)
    assert [r["status"] for r in raw] == ["fitted", "fitted", "fitted", "reused"]
    assert len(models) == 2 and models[0] is not models[1]
    assert result.final_trials[-1].reused
    assert not result.final_trials[0].reused


def test_changed_frozen_bytes_rejected_before_native_fit(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crf.models import training

    confidence(monkeypatch, 1.0)
    original = pipeline.freeze_datasets

    def tamper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = result[0].path
        path.write_bytes(
            path.read_bytes().replace(b'"schema_version":1', b'"schema_version": 1')
        )

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("native CRF must not be constructed for changed bytes")

        monkeypatch.setattr(training, "CRF", forbidden)
        return result

    monkeypatch.setattr(pipeline, "freeze_datasets", tamper)
    with pytest.raises(
        pipeline.DevelopmentIncomplete, match="identity/counts changed"
    ) as caught:
        pipeline.search_development(config_for(artifact_config), development_repository)
    assert ledger(development_repository, caught.value.run_id)[1]["status"] == "failed"


def test_preparation_resource_failure_keeps_plan(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MemoryMonitor.__exit__

    def fail(self: MemoryMonitor, *args: Any) -> Any:
        original(self, *args)
        raise MemoryBudgetExceeded("preparation resource exit")

    monkeypatch.setattr(MemoryMonitor, "__exit__", fail)
    with pytest.raises(pipeline.DevelopmentIncomplete, match="resource exit") as caught:
        pipeline.search_development(config_for(artifact_config), development_repository)
    assert all(
        t["status"] == "not_executed"
        for t in ledger(development_repository, caught.value.run_id)
    )


def test_failure_after_one_final_fit_preserves_outcomes(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confidence(monkeypatch, 1.0)
    original = pipeline._fit_final
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second final failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_fit_final", fail_second)
    with pytest.raises(pipeline.DevelopmentIncomplete) as caught:
        pipeline.search_development(config_for(artifact_config), development_repository)
    assert [
        t["status"] for t in ledger(development_repository, caught.value.run_id)
    ] == ["fitted", "fitted", "failed", "not_executed"]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["augmented", "empty", "mixed"])
def test_frozen_development_and_read_only_reload(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from crf.experiments import load_development
    from crf.models.bundles import parse_metadata
    from crf.utils.serialization import config_sha256

    config = config_for(artifact_config, full=mode != "mixed")
    confidence(monkeypatch, {"augmented": 1.0, "empty": 0.1, "mixed": 0.95}[mode])
    result = pipeline.develop(config, development_repository)
    raw = json.loads(result.run_path.read_bytes())
    assert raw["status"] == "complete" and raw["error"] is None
    assert raw["effective_config_sha256"] == config_sha256(config)
    assert raw["started_at"] <= raw["finished_at"]
    assert raw["started_at"].endswith("Z")
    assert raw["peak_rss_mib"] > 0
    assert len(ledger(development_repository, result.run_id)) == (
        4 if mode == "mixed" else 36
    )
    role_metadata = []
    for reference in (
        result.selection.gate,
        result.selection.final,
        result.selection.gold_control,
    ):
        path = (
            development_repository
            / "models"
            / result.run_id
            / "candidates"
            / reference.candidate_id
            / "metadata.json"
        )
        role_metadata.append(parse_metadata(json.loads(path.read_bytes())))
    gate, final_model, control = role_metadata
    assert gate.training.pseudo_token_count == control.training.pseudo_token_count == 0
    assert final_model.training.crf_parameters == control.training.crf_parameters
    if mode == "empty":
        assert (
            result.selection.gate
            == result.selection.final
            == result.selection.gold_control
        )
        assert result.selection.selected_threshold == 0.98
    if mode == "augmented":
        assert final_model.training.pseudo_token_count > 0
        assert result.selection.selected_threshold == 0.98
    snapshot = {
        path: path.read_bytes()
        for tree in ("models", "reports", "data/processed")
        for path in (development_repository / tree).rglob("*")
        if path.is_file()
    }
    restored = load_development(development_repository, result.run_id)
    assert restored == result
    assert snapshot == {
        path: path.read_bytes()
        for tree in ("models", "reports", "data/processed")
        for path in (development_repository / tree).rglob("*")
        if path.is_file()
    }


@pytest.fixture
def frozen_case(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    confidence(monkeypatch, 0.1)
    config = config_for(artifact_config)
    result = pipeline.develop(config, development_repository)
    return development_repository, config, result


@pytest.mark.parametrize(
    "fault",
    [
        "run_unknown",
        "run_bool_peak",
        "run_reverse_time",
        "run_duplicate",
        "run_bool_schema",
        "run_bad_environment",
        "source_dirty_type",
        "selected_role",
        "selected_threshold",
        "selected_unknown",
        "trials_missing_row",
        "trial_failed",
        "trial_wrong_reuse",
        "trial_bool_metric",
        "metrics",
        "dataset",
        "model",
        "metadata",
        "config",
        "inputs",
        "raw",
        "missing_model",
        "missing_completion",
        "failed_completion",
        "missing_trials",
    ],
)
def test_frozen_loader_refuses_changed_history(frozen_case: Any, fault: str) -> None:
    import hashlib

    from crf.experiments import load_development
    from crf.utils.serialization import ValidationError

    root, config, result = frozen_case
    report = root / "reports" / result.run_id
    models = root / "models" / result.run_id
    descriptor = json.loads(result.run_path.read_bytes())
    if fault.startswith("run_") or fault == "source_dirty_type":
        if fault == "run_duplicate":
            encoded = result.run_path.read_bytes()
            result.run_path.write_bytes(b'{"schema_version":1,' + encoded[1:])
        else:
            field, value = {
                "run_unknown": ("unknown", 1),
                "run_bool_peak": ("peak_rss_mib", True),
                "run_reverse_time": ("finished_at", "2000-01-01T00:00:00Z"),
                "run_bool_schema": ("schema_version", True),
                "run_bad_environment": ("dependency_versions", {}),
                "source_dirty_type": ("source_dirty", 1),
            }[fault]
            descriptor[field] = value
            result.run_path.write_text(json.dumps(descriptor), encoding="utf-8")
    elif fault.startswith("selected_"):
        path = models / "selected.json"
        selected = json.loads(path.read_bytes())
        if fault == "selected_role":
            selected["final"]["model_sha256"] = "f" * 64
        elif fault == "selected_threshold":
            selected["selected_threshold"] = 0.9
        else:
            selected["unknown"] = 1
        path.write_text(json.dumps(selected), encoding="utf-8")
        descriptor["selected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.run_path.write_text(json.dumps(descriptor), encoding="utf-8")
    elif fault.startswith("trial"):
        path = report / "trials.jsonl"
        rows = ledger(root, result.run_id)
        if fault == "trials_missing_row":
            rows.pop()
        elif fault == "trial_failed":
            rows[-1].update(
                status="failed",
                model_ref=None,
                validation_accuracy=None,
                validation_macro_f1=None,
                error=dict(stage="fit", message="failure"),
            )
        elif fault == "trial_wrong_reuse":
            rows[-1]["model_ref"]["candidate_id"] = "baseline_missing"
        else:
            rows[-1]["validation_accuracy"] = True
        path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        descriptor["trials_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.run_path.write_text(json.dumps(descriptor), encoding="utf-8")
    elif fault == "metrics":
        path = report / "validation/baseline_000/metrics.json"
        raw = json.loads(path.read_bytes())
        raw["run_id"] = "other_run"
        path.write_text(json.dumps(raw), encoding="utf-8")
    elif fault == "dataset":
        path = root / "data/processed" / result.run_id / "extended_000.jsonl"
        path.write_bytes(
            path.read_bytes().replace(b'"schema_version":1', b'"schema_version": 1')
        )
    elif fault in ("model", "metadata", "missing_model"):
        path = (
            models
            / "candidates/baseline_000"
            / ("metadata.json" if fault == "metadata" else "model.joblib")
        )
        if fault == "missing_model":
            path.unlink()
        else:
            path.write_bytes(b"changed")
    elif fault == "config":
        path = report / "effective_config.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "inputs":
        path = report / "inputs.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["source_sha256"] = "f" * 64
        path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        descriptor["inputs_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.run_path.write_text(json.dumps(descriptor), encoding="utf-8")
    elif fault == "raw":
        (root / "data/raw/literature/а.txt").write_text("Інше", encoding="utf-8")
    elif fault == "missing_completion":
        result.run_path.unlink()
    elif fault == "missing_trials":
        (report / "trials.jsonl").unlink()
    else:
        descriptor.update(
            status="failed",
            selected_sha256=None,
            error=dict(stage="freeze", message="failure"),
        )
        result.run_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises((OSError, ValidationError)):
        load_development(root, result.run_id)


def test_failed_search_has_failed_completion(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crf.experiments import load_development

    confidence(monkeypatch, 1.0)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("final fit controlled failure")

    monkeypatch.setattr(pipeline, "fit_candidate", fail)
    with pytest.raises(pipeline.DevelopmentIncomplete) as caught:
        pipeline.develop(config_for(artifact_config), development_repository)
    path = development_repository / "reports" / caught.value.run_id / "run.json"
    raw = json.loads(path.read_bytes())
    assert raw["status"] == "failed" and raw["selected_sha256"] is None
    assert len(ledger(development_repository, caught.value.run_id)) == 4
    with pytest.raises(ValueError):
        load_development(development_repository, caught.value.run_id)


def test_outer_monitor_failure_cannot_authorize_selection(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confidence(monkeypatch, 0.1)

    class FailingExit(MemoryMonitor):
        def __exit__(self, *args: Any) -> Any:
            super().__exit__(*args)
            raise MemoryBudgetExceeded("outer exit controlled failure")

    created = 0

    def factory(budget: int) -> MemoryMonitor:
        nonlocal created
        created += 1
        cls = FailingExit if created == 1 else MemoryMonitor
        return cls(budget, sampler=lambda: 1, interval_seconds=60)

    monkeypatch.setattr(pipeline, "MemoryMonitor", factory)
    with pytest.raises(pipeline.DevelopmentIncomplete, match="outer exit") as caught:
        pipeline.develop(config_for(artifact_config), development_repository)
    run = json.loads(
        (
            development_repository / "reports" / caught.value.run_id / "run.json"
        ).read_bytes()
    )
    assert run["status"] == "failed" and run["selected_sha256"] is None
    assert not (
        development_repository / "models" / caught.value.run_id / "selected.json"
    ).exists()


@pytest.mark.parametrize("collision", ["selected", "completion"])
def test_frozen_publication_preserves_existing_files(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    confidence(monkeypatch, 0.1)
    original = pipeline.search_development
    existing = []

    def collide(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = (
            development_repository
            / ("models" if collision == "selected" else "reports")
            / result.run_id
            / ("selected.json" if collision == "selected" else "run.json")
        )
        path.write_bytes(b"preserved")
        existing.append(path)
        return result

    monkeypatch.setattr(pipeline, "search_development", collide)
    with pytest.raises(pipeline.DevelopmentIncomplete):
        pipeline.develop(config_for(artifact_config), development_repository)
    assert existing[0].read_bytes() == b"preserved"


def test_source_provenance_is_observational(
    development_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crf.experiments import load_development

    confidence(monkeypatch, 0.1)
    original = pipeline._context
    monkeypatch.setattr(
        pipeline,
        "_context",
        lambda root: replace(
            original(root), source_revision="a" * 40, source_dirty=True
        ),
    )
    result = pipeline.develop(config_for(artifact_config), development_repository)
    raw = json.loads(result.run_path.read_bytes())
    assert raw["source_revision"] == "a" * 40 and raw["source_dirty"] is True
    assert (
        load_development(development_repository, result.run_id).selection
        == result.selection
    )
