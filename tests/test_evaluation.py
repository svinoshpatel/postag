"""Deterministic, manually calculated validation fixtures; no model or corpus IO."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from crf.data.records import SentenceRecord
from crf.evaluation import (
    ModelReference,
    SentencePrediction,
    gold_vocabulary,
    publish_validation_report,
    score_validation,
    validate_metric_report,
)
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryMonitor, ResourceMonitoringError
from crf.utils.serialization import ValidationError, loads_json


def sentence(
    config: PipelineConfig,
    labels: tuple[str, ...],
    forms: tuple[str, ...],
    role: str = "val_gold",
    ordinal: int = 1,
) -> SentenceRecord:
    return SentenceRecord(
        1,
        "gold",
        str(ordinal) * 64,
        getattr(config.data, role).as_posix(),
        "a" * 64,
        role,
        ordinal,
        None,
        " ".join(forms),
        "original",
        tuple(range(1, len(forms) + 1)),
        forms,
        forms,
        (None,) * len(forms),
        labels,
        config.label_space.map_upos(labels),
        None,
    )


def prediction(record: SentenceRecord, labels: tuple[str, ...]) -> SentencePrediction:
    return SentencePrediction(
        record.sentence_id, record.token_ids, record.feature_forms, labels
    )


REF = ModelReference("candidate", "a" * 64, "b" * 64)


def score(
    config: PipelineConfig, gold: list[SentenceRecord], labels: list[tuple[str, ...]]
) -> dict[str, object]:
    training = sentence(config, ("NOUN",), ("Кіт",), "train_gold")
    return score_validation(
        gold,
        [prediction(r, p) for r, p in zip(gold, labels, strict=True)],
        config=config,
        vocabulary=gold_vocabulary([training], config),
        model_ref=REF,
    )


def report(config: PipelineConfig, metrics: dict[str, object]) -> dict[str, object]:
    return dict(
        schema_version=1,
        run_id="run",
        split="val_gold",
        target_inventory=list(config.target_inventory),
        models={"trial": metrics},
    )


def test_manual_pooled_metrics(artifact_config: PipelineConfig) -> None:
    c = artifact_config
    gold = [
        sentence(c, ("NOUN", "VERB", "PUNCT"), ("Кіт", "спить", ".")),
        sentence(c, ("NOUN",), ("кіт",), ordinal=2),
    ]
    result = score(c, gold, [("NOUN", "NOUN", "PUNCT"), ("VERB",)])
    assert result["accuracy"] == 0.5  # Not mean sentence accuracy (1/3).
    assert result["token_count"] == 4 and result["sentence_count"] == 2
    assert result["per_tag"]["NOUN"] == dict(
        precision=0.5, recall=0.5, f1=0.5, support=2
    )
    assert result["per_tag"]["VERB"] == dict(
        precision=0.0, recall=0.0, f1=0.0, support=1
    )
    assert result["macro_f1"] == pytest.approx(1.5 / 17)
    assert result["oov"] == dict(token_count=2, correct_token_count=1, accuracy=0.5)
    assert result["non_punctuation"] == dict(
        token_count=3, correct_token_count=1, accuracy=1 / 3
    )
    labels = c.target_inventory
    assert result["confusion_matrix"][labels.index("VERB")][labels.index("NOUN")] == 1
    validate_metric_report(report(c, result), c)


@pytest.mark.parametrize("mode", ["empty_oov", "empty_nonpunct", "mapped", "symbols"])
def test_diagnostic_and_inventory_boundaries(
    artifact_config: PipelineConfig, mode: str
) -> None:
    c = artifact_config
    if mode == "mapped":
        c = replace(
            c,
            tag_mapping=tuple(
                (tag, "FUNC") for tag in ("ADP", "CCONJ", "PART", "SCONJ")
            ),
        )
        gold = sentence(c, ("ADP",), ("У",))
        result = score(c, [gold], [("FUNC",)])
        assert result["macro_f1"] == pytest.approx(1 / 14)
    elif mode == "empty_oov":
        result = score(c, [sentence(c, ("NOUN",), ("КІТ",))], [("NOUN",)])
        assert result["oov"] == dict(
            token_count=0, correct_token_count=0, accuracy=None
        )
    elif mode == "empty_nonpunct":
        result = score(c, [sentence(c, ("PUNCT",), ("!",))], [("NOUN",)])
        assert result["non_punctuation"] == dict(
            token_count=0, correct_token_count=0, accuracy=None
        )
    else:
        result = score(c, [sentence(c, ("SYM", "X"), ("+", "???"))], [("SYM", "X")])
        assert result["token_count"] == 2 and result["accuracy"] == 1
    validate_metric_report(report(c, result), c)


@pytest.mark.parametrize(
    "mode",
    [
        "empty",
        "missing",
        "extra",
        "duplicate",
        "identity",
        "ids",
        "forms",
        "length",
        "unknown",
        "test",
    ],
)
def test_alignment_failures(artifact_config: PipelineConfig, mode: str) -> None:
    c = artifact_config
    gold = sentence(c, ("NOUN",), ("Кіт",))
    p = prediction(gold, ("NOUN",))
    records, predictions = [gold], [p]
    if mode == "empty":
        records, predictions = [], []
    elif mode == "missing":
        predictions = []
    elif mode == "extra":
        predictions.append(p)
    elif mode == "duplicate":
        records, predictions = [gold, gold], [p, p]
    elif mode == "identity":
        predictions = [replace(p, sentence_id="wrong")]
    elif mode == "ids":
        predictions = [replace(p, token_ids=(2,))]
    elif mode == "forms":
        predictions = [replace(p, normalized_forms=("Пес",))]
    elif mode == "length":
        predictions = [replace(p, predicted_labels=())]
    elif mode == "unknown":
        predictions = [replace(p, predicted_labels=("INVALID",))]
    else:
        records = [replace(gold, source_role="test_gold")]
    vocabulary = gold_vocabulary([sentence(c, ("NOUN",), ("Кіт",), "train_gold")], c)
    with pytest.raises(ValueError):
        score_validation(
            records, predictions, config=c, vocabulary=vocabulary, model_ref=REF
        )


@pytest.mark.parametrize("role", ["val_gold", "test_gold", "literature"])
def test_vocab_rejects_nontraining(artifact_config: PipelineConfig, role: str) -> None:
    record = sentence(artifact_config, ("NOUN",), ("Кіт",), "train_gold")
    with pytest.raises(ValueError):
        gold_vocabulary([replace(record, source_role=role)], artifact_config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("accuracy", True),
        ("accuracy", float("nan")),
        ("accuracy", 0.5),
        ("macro_f1", 1.0),
        ("correct_token_count", True),
        ("sentence_count", 0),
        ("token_count", 2),
        ("extra", None),
        ("confusion_matrix", [[1]]),
    ],
)
def test_report_rejects_inconsistent_values(
    artifact_config: PipelineConfig, field: str, value: object
) -> None:
    c = artifact_config
    metrics = score(c, [sentence(c, ("NOUN",), ("Кіт",))], [("NOUN",)])
    metrics[field] = value
    with pytest.raises(ValueError):
        validate_metric_report(report(c, metrics), c)


def test_report_publication_and_tolerance(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    c = artifact_config
    metrics = score(c, [sentence(c, ("NOUN",), ("Кіт",))], [("NOUN",)])
    with MemoryMonitor(8192, sampler=lambda: 1, interval_seconds=60) as monitor:
        pass
    path = publish_validation_report(
        metrics,
        config=c,
        repository_root=artifact_repository,
        run_id="run",
        trial_id="trial",
        monitor=monitor,
    )
    assert loads_json(path.read_bytes()) == report(c, metrics)
    with pytest.raises(ValueError):
        publish_validation_report(
            metrics,
            config=c,
            repository_root=artifact_repository,
            run_id="run",
            trial_id="trial",
            monitor=monitor,
        )
    value = copy.deepcopy(report(c, metrics))
    value["models"]["trial"]["accuracy"] -= 1e-14
    validate_metric_report(value, c)
    with pytest.raises(ResourceMonitoringError):
        publish_validation_report(
            metrics,
            config=c,
            repository_root=artifact_repository,
            run_id="other",
            trial_id="trial",
            monitor=MemoryMonitor(8192, sampler=lambda: 1),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"token_count": True},
        {"token_count": 2},
        {"correct_token_count": 1},
        {"accuracy": 0.0},
        {"extra": 0},
    ],
)
def test_invalid_empty_diagnostic(
    artifact_config: PipelineConfig, change: dict[str, object]
) -> None:
    c = artifact_config
    metrics = score(c, [sentence(c, ("NOUN",), ("Кіт",))], [("NOUN",)])
    metrics["oov"].update(change)
    with pytest.raises(ValidationError):
        validate_metric_report(report(c, metrics), c)


def test_vocabulary_reuse_and_configuration_binding(
    artifact_config: PipelineConfig,
) -> None:
    c = artifact_config
    training = sentence(c, ("NOUN",), ("Кіт",), "train_gold")
    vocabulary = gold_vocabulary([training], c)
    gold = sentence(c, ("NOUN",), ("Пес",))
    for predicted in (("NOUN",), ("VERB",)):
        result = score_validation(
            [gold],
            [prediction(gold, predicted)],
            config=c,
            vocabulary=vocabulary,
            model_ref=REF,
        )
        assert result["oov"]["token_count"] == 1
    with pytest.raises(ValueError):
        score_validation(
            [gold],
            [prediction(gold, ("NOUN",))],
            config=replace(c, seed=1),
            vocabulary=vocabulary,
            model_ref=REF,
        )
    with pytest.raises(ValueError):
        gold_vocabulary([], c)
