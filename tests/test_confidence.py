"""Offline confidence contract fixtures; no production data or network."""

from typing import Any

import pytest

from crf.models.confidence import (
    SentenceConfidence,
    iter_confidence_candidates,
    score_sentence,
    viterbi_confidence,
)
from crf.utils.config import PipelineConfig
from crf.utils.serialization import ValidationError


def test_whole_sentence_and_exact_boundary() -> None:
    score = SentenceConfidence(("NOUN", "VERB", "X"), (0.98, 0.95, 0.87))
    assert not score.accepts(0.90)
    assert score.accepts(0.87)
    assert score.min_confidence == 0.87
    assert not SentenceConfidence(("X",), (0.8999999999999999,)).accepts(0.9)
    assert not SentenceConfidence((), ()).accepts(0.0)


def test_actual_viterbi_label_and_custom_inventory() -> None:
    score = viterbi_confidence(
        ("SERVICE", "NOUN"),
        ({"SERVICE": 0.4, "NOUN": 0.6}, {"SERVICE": 0.1, "NOUN": 0.9}),
        target_inventory=("SERVICE", "NOUN"),
    )
    assert score.labels == ("SERVICE", "NOUN")
    assert score.token_confidences == (0.4, 0.9)
    assert not score.accepts(0.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.01, True])
def test_invalid_probabilities_fail(value: Any) -> None:
    with pytest.raises(ValidationError):
        SentenceConfidence(("X",), (value,))
    with pytest.raises(ValidationError):
        SentenceConfidence(("X",), (1.0,)).accepts(value)
    with pytest.raises(ValidationError):
        viterbi_confidence(
            ("X",), ({"X": 1.0, "NOUN": value},), target_inventory=("X", "NOUN")
        )


@pytest.mark.parametrize(
    ("labels", "rows"),
    [
        (("X",), []),
        (("X",), [{}]),
        (("OTHER",), [{"OTHER": 1.0}]),
        (("X",), [{"X": 1.0, "OTHER": 0.0}]),
    ],
)
def test_bad_model_outputs(labels: Any, rows: Any) -> None:
    with pytest.raises(ValidationError):
        viterbi_confidence(labels, rows, target_inventory=("X",))


class FakeGate:
    def predict_single(self, features: Any) -> list[str]:
        return ["NOUN"] * len(features)

    def predict_marginals_single(self, features: Any) -> list[dict[str, float]]:
        return [{"NOUN": 0.95}] * len(features)


def test_scoring_and_empty(artifact_config: PipelineConfig) -> None:
    score = score_sentence(("Кіт", "спить"), model=FakeGate(), config=artifact_config)
    assert score.labels == ("NOUN", "NOUN")
    assert score.token_confidences == (0.95, 0.95)
    assert score_sentence((), model=None, config=artifact_config).labels == ()


def test_streaming_pool(
    artifact_config: PipelineConfig, artifact_repository: Any
) -> None:
    from crf.data.literature import RawSentence, iter_literature

    sentences = (
        s
        for s in iter_literature(artifact_config, artifact_repository)
        if isinstance(s, RawSentence)
    )
    candidates = list(
        iter_confidence_candidates(sentences, model=FakeGate(), config=artifact_config)
    )
    assert candidates
    for candidate in candidates:
        assert len(candidate.confidence.labels) == len(candidate.sentence.feature_forms)
        assert candidate.confidence.accepts(0.90)
        assert candidate.confidence.accepts(0.95)
        assert not candidate.confidence.accepts(0.98)


def test_lazy_pool_and_alignment(
    artifact_config: PipelineConfig, artifact_repository: Any
) -> None:
    from crf.data.literature import RawSentence, iter_literature

    raw = next(
        s
        for s in iter_literature(artifact_config, artifact_repository)
        if isinstance(s, RawSentence)
    )

    def source() -> Any:
        yield raw
        raise RuntimeError("late source failure")

    pool = iter_confidence_candidates(
        source(), model=FakeGate(), config=artifact_config
    )
    first = next(pool)
    assert first.sentence is raw
    with pytest.raises(RuntimeError, match="late source"):
        next(pool)

    class BadGate(FakeGate):
        def predict_single(self, features: Any) -> list[str]:
            return []

    with pytest.raises(ValidationError, match="every token"):
        score_sentence(raw.feature_forms, model=BadGate(), config=artifact_config)


@pytest.mark.integration
def test_native_viterbi_marginals(artifact_config: PipelineConfig) -> None:
    """Requires locked sklearn-crfsuite; tiny in-memory synthetic fit only."""
    import sklearn_crfsuite

    from crf.features.surface import sentence_features
    from crf.models.bundles import _dispose

    forms = ("Кіт", "спить")
    features = sentence_features(forms)
    model = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=10)
    try:
        model.fit([features], [["NOUN", "VERB"]])
        result = score_sentence(forms, model=model, config=artifact_config)
        predicted = tuple(model.predict_single(features))
        marginals = model.predict_marginals_single(features)
        assert result.labels == predicted
        assert result.token_confidences == tuple(
            row[label] for row, label in zip(marginals, predicted, strict=True)
        )
    finally:
        _dispose(model)
