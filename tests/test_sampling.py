"""Small deterministic sampling fixtures; no network or production inputs."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crf.data.identity import InputIdentity, sentence_id
from crf.data.literature import RawSentence
from crf.data.records import SentenceRecord
from crf.data.segmentation import AdaptedSentence, segmentation_metadata
from crf.models import sampling
from crf.models.confidence import ConfidenceCandidate, SentenceConfidence
from crf.models.sampling import freeze_datasets
from crf.utils.config import PipelineConfig
from crf.utils.paths import PathValidationError
from crf.utils.resources import MemoryBudgetExceeded, MemoryMonitor
from crf.utils.serialization import ValidationError, jsonl_bytes


@pytest.fixture(autouse=True)
def controlled_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sampling,
        "MemoryMonitor",
        lambda budget: MemoryMonitor(budget, sampler=lambda: 1, interval_seconds=60),
    )


def gold(config: PipelineConfig, ordinal: int = 1) -> SentenceRecord:
    path = config.data.train_gold.as_posix()
    return SentenceRecord(
        1,
        "gold",
        sentence_id("train_gold", path, "a" * 64, ordinal),
        path,
        "a" * 64,
        "train_gold",
        ordinal,
        None,
        "Кіт спить",
        "original",
        (1, 2),
        ("Кіт", "спить"),
        ("Кіт", "спить"),
        (None, None),
        ("NOUN", "VERB"),
        ("NOUN", "VERB"),
        None,
    )


def candidate(text: str, ordinal: int, confidence: float = 0.96) -> ConfidenceCandidate:
    source = InputIdentity(
        1, "literature", "data/raw/literature/example.txt", "b" * 64, 1000, "utf-8"
    )
    forms = tuple(text.split())
    offset = 0
    spans = []
    for form in forms:
        spans.append((offset, offset + len(form)))
        offset += len(form) + 1
    raw = RawSentence(
        source,
        sentence_id(
            source.source_role, source.source_path, source.source_sha256, ordinal
        ),
        1,
        ordinal,
        AdaptedSentence(text, (0, len(text)), forms, forms, tuple(spans)),
        segmentation_metadata(),
    )
    return ConfidenceCandidate(
        raw, SentenceConfidence(("NOUN",) * len(forms), (confidence,) * len(forms))
    )


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def freeze(
    config: PipelineConfig,
    root: Path,
    pool: Any,
    run: str = "sample",
    train: Any = None,
) -> Any:
    return freeze_datasets(
        [gold(config), gold(config, 2)] if train is None else train,
        pool,
        config=config,
        repository_root=root,
        run_id=run,
        gate_model_sha256="c" * 64,
    )


def priority(item: ConfidenceCandidate, seed: int) -> str:
    fingerprint = hashlib.sha256(item.sentence.text.surface_text.encode()).hexdigest()
    return hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest()


def test_order_thresholds_reuse_and_permutation(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    pool = [
        candidate("Кіт", 1, 0.90),
        candidate("Сонце", 2, 0.95),
        candidate("Дім", 3, 0.98),
    ]
    first = freeze(artifact_config, artifact_repository, iter(pool))
    second = freeze(artifact_config, artifact_repository, reversed(pool), "other")
    for a, b in zip(first, second, strict=True):
        assert a.sha256 == b.sha256 == hashlib.sha256(a.path.read_bytes()).hexdigest()
        assert a.path.read_bytes() == b.path.read_bytes()
        actual = rows(a.path)
        assert actual[:2] == [
            gold(artifact_config).to_dict(),
            gold(artifact_config, 2).to_dict(),
        ]
        expected = sorted(
            (c for c in pool if c.confidence.accepts(a.threshold)),
            key=lambda c: priority(c, artifact_config.seed),
        )
        assert [r["surface_text"] for r in actual[2:]] == [
            c.sentence.text.surface_text for c in expected
        ]
        for record in actual[2:]:
            assert record["pseudo"]["gate_model_sha256"] == "c" * 64
    assert [d.pseudo_sentences for d in first] == [3, 2, 1]


@pytest.mark.parametrize("ratio,expected", [(0.0, 0), (0.25, 1), (0.5, 1), (1.0, 2)])
def test_budget_and_underfill(
    artifact_config: PipelineConfig,
    artifact_repository: Path,
    ratio: float,
    expected: int,
) -> None:
    config = replace(
        artifact_config,
        self_training=replace(
            artifact_config.self_training, pseudo_to_gold_ratio=ratio
        ),
    )
    result = freeze(
        config,
        artifact_repository,
        [candidate("Кіт спить", 1, 1.0), candidate("Дім", 2, 1.0)],
    )
    for dataset in result:
        assert dataset.pseudo_tokens <= dataset.token_cap
        assert dataset.pseudo_sentences == expected
        assert (
            sum(len(r["token_ids"]) for r in rows(dataset.path)[2:])
            == dataset.pseudo_tokens
        )


def test_skip_oversized_continue(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    big = candidate("Кіт спить біля дому вранці", 1, 1.0)
    small = candidate("Дім", 2, 1.0)
    result = freeze(artifact_config, artifact_repository, [big, small])
    assert all(d.pseudo_tokens == 1 for d in result)
    assert rows(result[0].path)[-1]["surface_text"] == "Дім"


def test_empty_and_no_clobber(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    result = freeze(artifact_config, artifact_repository, [])
    expected = jsonl_bytes(gold(artifact_config).to_dict()) + jsonl_bytes(
        gold(artifact_config, 2).to_dict()
    )
    assert all(d.path.read_bytes() == expected and d.pseudo_tokens == 0 for d in result)
    with pytest.raises((FileExistsError, PathValidationError)):
        freeze(artifact_config, artifact_repository, [])
    assert result[0].path.read_bytes() == expected


@pytest.mark.parametrize("mode", ["duplicate", "late", "empty_gold", "heldout"])
def test_failures_no_dataset(
    artifact_config: PipelineConfig, artifact_repository: Path, mode: str
) -> None:
    c = candidate("Кіт", 1)

    def late() -> Any:
        yield c
        raise RuntimeError("late input")

    pool = (
        [c, candidate("Кіт", 2)]
        if mode == "duplicate"
        else late()
        if mode == "late"
        else []
    )
    train = (
        []
        if mode == "empty_gold"
        else [replace(gold(artifact_config), source_role="val_gold")]
        if mode == "heldout"
        else None
    )
    with pytest.raises((ValidationError, RuntimeError)):
        freeze(artifact_config, artifact_repository, pool, train=train)
    assert not (artifact_repository / "data/processed/sample").exists()


def test_resource_failure(
    artifact_config: PipelineConfig,
    artifact_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: MemoryMonitor) -> None:
        raise MemoryBudgetExceeded("controlled budget failure")

    monkeypatch.setattr(MemoryMonitor, "checkpoint", fail)
    with pytest.raises(MemoryBudgetExceeded):
        freeze(artifact_config, artifact_repository, [])
    assert not (artifact_repository / "data/processed/sample").exists()


def test_decimal_cap_and_full_precision(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    config = replace(
        artifact_config,
        self_training=replace(artifact_config.self_training, pseudo_to_gold_ratio=0.3),
    )
    result = freeze(
        config,
        artifact_repository,
        [candidate("Кіт", 1, 0.9499999999999999)],
        train=[gold(config, i) for i in range(1, 6)],
    )
    assert result[0].token_cap == 3
    assert [d.pseudo_tokens for d in result] == [1, 0, 0]


def test_seed_changes_priority(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    pool = [
        candidate(text, i) for i, text in enumerate(("Кіт", "Сонце", "Дім", "Небо"), 1)
    ]
    config = replace(artifact_config, seed=314)
    result = freeze(config, artifact_repository, pool)
    expected = sorted(pool, key=lambda c: priority(c, 314))
    assert [r["surface_text"] for r in rows(result[0].path)[2:]] == [
        c.sentence.text.surface_text for c in expected
    ]


def test_existing_dataset_preserved(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    path = artifact_repository / "data/processed/sample/extended_000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing")
    with pytest.raises((PathValidationError, FileExistsError)):
        freeze(artifact_config, artifact_repository, [])
    assert path.read_bytes() == b"existing"


def test_publication_monitor_exit_failure(
    artifact_config: PipelineConfig,
    artifact_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MemoryMonitor.__exit__
    exits = 0

    def fail(self: MemoryMonitor, *args: Any) -> Any:
        nonlocal exits
        result = original(self, *args)
        exits += 1
        if exits == 2:
            raise MemoryBudgetExceeded("publication exit")
        return result

    monkeypatch.setattr(MemoryMonitor, "__exit__", fail)
    with pytest.raises(MemoryBudgetExceeded, match="publication exit"):
        freeze(artifact_config, artifact_repository, [])
    assert not (
        artifact_repository / "data/processed/sample/extended_000.jsonl"
    ).exists()


def test_remaining_budget_skip_and_continue(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    pool = [
        candidate("Кіт спить", 1, 1.0),
        candidate("Сонце світить ясно", 2, 1.0),
        candidate("Дім", 3, 1.0),
    ]
    seed = next(
        s for s in range(100) if sorted(pool, key=lambda c: priority(c, s)) == pool
    )
    config = replace(artifact_config, seed=seed)
    result = freeze(config, artifact_repository, reversed(pool))
    assert all(d.token_cap == 4 and d.pseudo_tokens == 3 for d in result)
    assert [r["surface_text"] for r in rows(result[0].path)[2:]] == ["Кіт спить", "Дім"]


def test_manual_diagnostics_full_stream(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    from crf.models.sampling import freeze_datasets

    config = replace(
        artifact_config,
        self_training=replace(artifact_config.self_training, pseudo_to_gold_ratio=0.25),
    )
    pool = [
        candidate("Кіт спить", 1, 0.8),
        candidate("Сонце світить", 2, 0.95),
        candidate("Дім", 3, 0.99),
    ]
    freeze_datasets(
        [gold(config), gold(config, 2)],
        pool,
        config=config,
        repository_root=artifact_repository,
        run_id="diagnostics",
        gate_model_sha256="c" * 64,
        all_scored=True,
    )
    report = json.loads(
        (artifact_repository / "reports/diagnostics/pseudo_selection.json").read_text()
    )
    assert report["confidence_denominator"] == "all_scored"
    assert report["upstream_audit"] is None
    low, middle, high = report["thresholds"]
    assert low["received"] == dict(sentences=3, tokens=5)
    assert low["eligible"] == middle["eligible"] == dict(sentences=2, tokens=3)
    assert low["selected"] == dict(sentences=1, tokens=1)
    assert low["confidence_excluded"] == dict(sentences=1, tokens=2)
    assert low["budget_excluded"] == dict(sentences=1, tokens=2)
    assert low["cap_utilization"] == 1.0
    assert low["by_predicted_pos"] == [
        dict(
            label="NOUN",
            received_tokens=5,
            eligible_tokens=3,
            selected_tokens=1,
            confidence_acceptance_rate=3 / 5,
        )
    ]
    assert sum(g["received_sentences"] for g in low["by_source"]) == 3
    assert {
        g["value"]: g["confidence_acceptance_rate"] for g in low["by_sentence_length"]
    } == {1: 1.0, 2: 0.5}
    assert high["eligible"] == high["selected"] == dict(sentences=1, tokens=1)


def test_limited_diagnostics_zero_budget(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    config = replace(
        artifact_config,
        self_training=replace(artifact_config.self_training, pseudo_to_gold_ratio=0.0),
    )
    freeze(config, artifact_repository, [candidate("Кіт", 1)])
    report = json.loads(
        (artifact_repository / "reports/sample/pseudo_selection.json").read_text()
    )
    assert report["confidence_denominator"] == "minimum_threshold_pool"
    assert all(
        t["cap_utilization"] is None and not t["augmentation_selected"]
        for t in report["thresholds"]
    )


def test_rejects_in_limited_pool_fail(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    with pytest.raises(ValidationError, match="all_scored"):
        freeze(artifact_config, artifact_repository, [candidate("Кіт", 1, 0.5)])


def test_full_stream_scoring_keeps_rejects(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    from crf.models.confidence import iter_confidence_candidates, iter_scored_candidates

    class Gate:
        def predict_single(self, features: Any) -> list[str]:
            return ["NOUN"] * len(features)

        def predict_marginals_single(self, features: Any) -> list[dict[str, float]]:
            return [{"NOUN": 0.5}] * len(features)

    raw = candidate("Кіт", 1).sentence
    assert (
        len(list(iter_scored_candidates([raw], model=Gate(), config=artifact_config)))
        == 1
    )
    assert (
        list(iter_confidence_candidates([raw], model=Gate(), config=artifact_config))
        == []
    )


@pytest.mark.parametrize("mode", ["config", "count", "valid"])
def test_audit_binding(
    artifact_config: PipelineConfig, artifact_repository: Path, mode: str
) -> None:
    from crf.data.contamination import AuditReport
    from crf.models.sampling import freeze_datasets
    from crf.utils.serialization import config_sha256

    audit = AuditReport(
        1,
        1,
        segmentation_metadata(),
        config_sha256(artifact_config) if mode != "config" else "f" * 64,
        (),
        (),
        2,
        0,
        1 if mode != "count" else 2,
        1,
        (("gold_overlap", 3),),
        0,
        0,
        "d" * 64,
    )

    def invoke() -> Any:
        return freeze_datasets(
            [gold(artifact_config)],
            [candidate("Кіт", 1)],
            config=artifact_config,
            repository_root=artifact_repository,
            run_id="audit",
            gate_model_sha256="c" * 64,
            all_scored=True,
            audit_report=lambda: audit,
        )

    if mode != "valid":
        with pytest.raises(ValidationError, match="Audit"):
            invoke()
    else:
        invoke()
        report = json.loads(
            (artifact_repository / "reports/audit/pseudo_selection.json").read_text()
        )
        assert report["upstream_audit"]["exclusion_counts"] == {"gold_overlap": 3}


def test_diagnostic_publication_no_clobber(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    path = artifact_repository / "reports/sample/pseudo_selection.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing diagnostics")
    with pytest.raises((PathValidationError, FileExistsError)):
        freeze(artifact_config, artifact_repository, [])
    assert path.read_bytes() == b"existing diagnostics"


def test_rejected_bodies_not_retained(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    import sqlite3

    sampling.freeze_datasets(
        [gold(artifact_config)],
        [candidate("Кіт", 1, 0.8)],
        config=artifact_config,
        repository_root=artifact_repository,
        run_id="reject",
        gate_model_sha256="c" * 64,
        all_scored=True,
    )
    with sqlite3.connect(
        artifact_repository / "data/interim/reject/sampling/candidates.sqlite3"
    ) as db:
        assert db.execute("SELECT body FROM pool").fetchall() == [(None,)]
