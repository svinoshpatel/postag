"""Feature expectations derived from the architecture, without corpus inputs."""

import pytest

from crf.features.surface import normalize_form, sentence_features


def test_features_and_short_affixes() -> None:
    features = sentence_features(["ҐАНОК", "гідність", "Я"])
    assert features[0]["word.lower"] == "ґанок"
    assert features[0]["is_uppercase"] is True
    assert features[0]["is_title"] is False
    assert features[1]["suffix.5"] == "ність"
    assert features[1]["prefix.4"] == "гідн"
    assert features[2]["prefix.1"] == "я"
    assert "prefix.2" not in features[2]
    assert "suffix.2" not in features[2]
    assert features[2]["bias"] == 1.0
    assert "word.tf" not in features[0]


def test_context_and_boundaries() -> None:
    features = sentence_features(["Київ", "—", "сім’я", "123", "."])
    assert features[2]["-2:word.lower"] == "київ"
    assert features[2]["+2:is_punct"] is True
    assert features[2]["+1:has_digits"] is True
    assert not any(
        "prefix" in key or "suffix" in key for key in features[2] if ":" in key
    )
    singleton = sentence_features(["ї"])[0]
    assert {key for key in singleton if ":" in key} == {
        "-2:BOS",
        "-1:BOS",
        "+1:EOS",
        "+2:EOS",
    }
    assert all(singleton[key] is True for key in singleton if ":" in key)


@pytest.mark.parametrize("apostrophe", ["'", "’", "`", "ʼ"])
def test_normalization(apostrophe: str) -> None:
    assert normalize_form(f"п{apostrophe}ять") == "п'ять"
    assert normalize_form("і\u0308 ҐгИиІіЕеЄєЇїь\u0301") == "ї ҐгИиІіЕеЄєЇїь\u0301"
    assert normalize_form("а\u00a0б\u202fв\u200b") == "а б в\u200b"


@pytest.mark.parametrize("hyphen", ["-", "‐", "‑"])
def test_shape_flags(hyphen: str) -> None:
    features = sentence_features([f"будь{hyphen}який", "²", "…!", "a."])
    assert features[0]["has_hyphen"] is True
    assert features[0]["is_lowercase"] is True
    assert features[1]["has_digits"] is True
    assert features[2]["is_punct"] is True
    assert features[3]["is_punct"] is False


def test_invalid_forms_and_annotations() -> None:
    for forms in ([], [""], ["Київ", ""]):
        with pytest.raises(ValueError):
            sentence_features(forms)
    with pytest.raises(ValueError):
        sentence_features("Київ")
    with pytest.raises(ValueError):
        sentence_features([("Київ", "PROPN")])  # type: ignore[list-item]


def test_unseen_forms() -> None:
    assert sentence_features(["невідомеслово"])[0]["word.lower"] == "невідомеслово"
