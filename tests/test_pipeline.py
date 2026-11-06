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

from crf import pipeline
from crf.models import bundles, reuse, sampling, search
from crf.models.confidence import ConfidenceCandidate, SentenceConfidence
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryBudgetExceeded, MemoryMonitor
from crf.utils.serialization import canonical_config_bytes


@pytest.fixture(autouse=True)
def controlled_monitor(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    log = logging.getLogger("crf")
    previous = (log.handlers[:], log.level, log.propagate)
    for module in (pipeline, bundles, reuse, sampling, search):
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
    result = pipeline.develop(config, development_repository)
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
    result = pipeline.develop(config, development_repository)
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
        pipeline.develop(config, development_repository)
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
        pipeline.develop(config_for(artifact_config), development_repository)
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
        pipeline.develop(config_for(artifact_config), development_repository)
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
    assert output["status"] == "search_complete_pending_freeze"
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
    result = pipeline.develop(config, development_repository)
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
        pipeline.develop(config_for(artifact_config), development_repository)
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
        pipeline.develop(config_for(artifact_config), development_repository)
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
        pipeline.develop(config_for(artifact_config), development_repository)
    assert [
        t["status"] for t in ledger(development_repository, caught.value.run_id)
    ] == ["fitted", "fitted", "failed", "not_executed"]
