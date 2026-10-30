"""Feature expectations derived from the architecture, without corpus inputs."""

import copy
import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from crf.data.identity import InputIdentity, sentence_id
from crf.data.records import parse_sentence_record
from crf.features.schema import (
    FEATURE_SCHEMA_VERSION,
    FeatureSchemaError,
    validate_feature_version,
    validate_sentence_features,
)
from crf.features.surface import normalize_form, sentence_features
from crf.utils.config import ConfigError, PipelineConfig, parse_config


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


@pytest.fixture
def singleton_dictionary() -> dict[str, object]:
    """Independent complete version-1 expectation for the surface form Кіт."""
    return {
        "bias": 1.0,
        "word.lower": "кіт",
        "prefix.1": "к",
        "prefix.2": "кі",
        "prefix.3": "кіт",
        "suffix.1": "т",
        "suffix.2": "іт",
        "suffix.3": "кіт",
        "is_title": True,
        "is_uppercase": False,
        "is_lowercase": False,
        "has_hyphen": False,
        "has_digits": False,
        "is_punct": False,
        "-2:BOS": True,
        "-1:BOS": True,
        "+1:EOS": True,
        "+2:EOS": True,
    }


def test_complete_singleton_and_validator(
    singleton_dictionary: dict[str, object],
) -> None:
    assert sentence_features(["Кіт"]) == [singleton_dictionary]
    assert validate_sentence_features([singleton_dictionary]) == [singleton_dictionary]
    assert FEATURE_SCHEMA_VERSION == validate_feature_version(1) == 1
    assert type(sentence_features(["Кіт"])[0]["bias"]) is float


def test_complete_center_context_dictionary() -> None:
    actual = sentence_features(["Я", "²", "—", "ҐАНОК", "x.y"])[2]
    expected: dict[str, object] = {
        "bias": 1.0,
        "word.lower": "—",
        "prefix.1": "—",
        "suffix.1": "—",
        "is_title": False,
        "is_uppercase": False,
        "is_lowercase": False,
        "has_hyphen": False,
        "has_digits": False,
        "is_punct": True,
    }
    # Each tuple is lexical/title/upper/lower/hyphen/digits/punctuation, from the spec.
    neighbors = {
        "-2:": ("я", True, True, False, False, False, False),
        "-1:": ("²", False, False, False, False, True, False),
        "+1:": ("ґанок", False, True, False, False, False, False),
        "+2:": ("x.y", False, False, True, False, False, False),
    }
    for prefix, values in neighbors.items():
        expected.update(
            zip(
                (
                    prefix + key
                    for key in (
                        "word.lower",
                        "is_title",
                        "is_uppercase",
                        "is_lowercase",
                        "has_hyphen",
                        "has_digits",
                        "is_punct",
                    )
                ),
                values,
                strict=True,
            )
        )
    assert actual == expected


@pytest.mark.parametrize(
    ("form", "prefixes", "suffixes"),
    [
        ("Я", ("я",), ("я",)),
        ("аб", ("а", "аб"), ("б", "аб")),
        ("абв", ("а", "аб", "абв"), ("в", "бв", "абв")),
        ("абвг", ("а", "аб", "абв", "абвг"), ("г", "вг", "бвг", "абвг")),
        ("абвгд", ("а", "аб", "абв", "абвг"), ("д", "гд", "вгд", "бвгд", "абвгд")),
        ("абвгде", ("а", "аб", "абв", "абвг"), ("е", "де", "где", "вгде", "бвгде")),
        ("І\u0308", ("ї",), ("ї",)),
        ("İ", ("i", "i\u0307"), ("\u0307", "i\u0307")),
        ("і\u0308\u0301", ("ї", "ї\u0301"), ("\u0301", "ї\u0301")),
    ],
)
def test_exact_affix_lengths_after_unicode_lexical_normalization(
    form: str,
    prefixes: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> None:
    features = sentence_features([form])[0]
    assert {
        key: value for key, value in features.items() if key.startswith("prefix.")
    } == {f"prefix.{index}": value for index, value in enumerate(prefixes, 1)}
    assert {
        key: value for key, value in features.items() if key.startswith("suffix.")
    } == {f"suffix.{index}": value for index, value in enumerate(suffixes, 1)}


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("Я", (True, True, False, False, False, False)),
        ("ҐАНОК", (False, True, False, False, False, False)),
        ("Київ", (True, False, False, False, False, False)),
        ("слово", (False, False, True, False, False, False)),
        ("iPhone", (False, False, False, False, False, False)),
        ("²", (False, False, False, False, True, False)),
        ("١٢", (False, False, False, False, True, False)),
        ("…!", (False, False, False, False, False, True)),
        ("—", (False, False, False, False, False, True)),
        ("а−б", (False, False, True, False, False, False)),
        ("_", (False, False, False, False, False, True)),
        ("€", (False, False, False, False, False, False)),
        ("a.", (False, False, True, False, False, False)),
    ],
)
def test_independent_unicode_shape_flags(form: str, expected: tuple[bool, ...]) -> None:
    features = sentence_features([form])[0]
    keys = (
        "is_title",
        "is_uppercase",
        "is_lowercase",
        "has_hyphen",
        "has_digits",
        "is_punct",
    )
    assert tuple(features[key] for key in keys) == expected
    assert all(type(features[key]) is bool for key in keys)


def test_every_short_sentence_boundary_and_context_limit() -> None:
    for size in range(1, 6):
        features = sentence_features(["я"] * size)
        for index, row in enumerate(features):
            for offset in (-2, -1, 1, 2):
                context = {key for key in row if key.startswith(f"{offset:+d}:")}
                if index + offset < 0:
                    assert context == {f"{offset:+d}:BOS"}
                elif index + offset >= size:
                    assert context == {f"{offset:+d}:EOS"}
                else:
                    assert context == {
                        f"{offset:+d}:{key}"
                        for key in (
                            "word.lower",
                            "is_title",
                            "is_uppercase",
                            "is_lowercase",
                            "has_hyphen",
                            "has_digits",
                            "is_punct",
                        )
                    }
    assert sentence_features(["Кіт"])[0]["+1:EOS"] is True
    assert sentence_features(["Пес"])[0]["-1:BOS"] is True
    left = sentence_features(["SECRET", "Я", "Я", "Кіт", "Я", "Я", "SECRET"])
    right = sentence_features(["different", "Я", "Я", "Кіт", "Я", "Я", "different"])
    assert left[3] == right[3]
    assert "secret" not in left[3].values()


@pytest.mark.parametrize(
    "forms",
    [
        None,
        {},
        {"Кіт": "NOUN"},
        iter(["Кіт"]),
        b"word",
        1,
        True,
        [None],
        [1],
        [{"form": "Кіт", "upos": "NOUN"}],
        [["Кіт", "NOUN"]],
        SimpleNamespace(feature_forms=("Кіт",), target_labels=("NOUN",)),
        ["\ud800"],
    ],
)
def test_invalid_forms_and_annotation_containers(forms: object) -> None:
    with pytest.raises(FeatureSchemaError):
        sentence_features(forms)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [0, 2, -1, True, False, 1.0, "1", None])
def test_versions_at_extractor_validator_and_config_boundaries(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    singleton_dictionary: dict[str, object],
    version: object,
) -> None:
    with pytest.raises(FeatureSchemaError, match="schema_version"):
        sentence_features(["Кіт"], schema_version=version)  # type: ignore[arg-type]
    with pytest.raises(FeatureSchemaError, match="schema_version"):
        validate_sentence_features([singleton_dictionary], schema_version=version)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="schema_version"):
        parse_config(
            artifact_config.to_dict() | {"features": {"schema_version": version}},
            repository_root=artifact_repository,
        )
    with pytest.raises(FeatureSchemaError, match="version"):
        sentence_features(["Кіт"], normalizer_version=version)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key",
    [
        "word.tf",
        "UPOS",
        "LEMMA",
        "FEATS",
        "source_path",
        "+3:word.lower",
        "-1:prefix.1",
        "+1:bias",
        "+1:word.lower",
        "prefix.4",
        "suffix.4",
    ],
)
def test_unknown_and_unavailable_keys_rejected(
    singleton_dictionary: dict[str, object],
    key: str,
) -> None:
    with pytest.raises(FeatureSchemaError, match="unsupported keys"):
        validate_sentence_features([singleton_dictionary | {key: "injected"}])


def test_all_required_keys_and_nonstrings_rejected(
    singleton_dictionary: dict[str, object],
) -> None:
    for key in singleton_dictionary:
        with pytest.raises(FeatureSchemaError, match="token 1"):
            validate_sentence_features(
                [
                    {
                        name: value
                        for name, value in singleton_dictionary.items()
                        if name != key
                    }
                ]
            )
    with pytest.raises(FeatureSchemaError, match="string keys"):
        validate_sentence_features([singleton_dictionary | {1: True}])


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("bias", 1),
        ("bias", True),
        ("bias", "1.0"),
        ("bias", None),
        ("bias", 0.0),
        ("bias", float("nan")),
        ("bias", float("inf")),
        ("bias", -float("inf")),
        ("is_title", 1),
        ("is_title", 1.0),
        ("has_digits", "False"),
        ("is_punct", None),
        ("word.lower", ""),
        ("word.lower", "Кіт"),
        ("word.lower", "і\u0308"),
        ("word.lower", "\ud800"),
        ("word.lower", False),
        ("word.lower", 1.0),
        ("prefix.1", 1),
        ("prefix.1", ""),
        ("prefix.1", "x"),
        ("suffix.2", "кі"),
        ("-1:BOS", False),
        ("+2:EOS", 1),
        ("+1:EOS", None),
    ],
)
def test_wrong_values_rejected_with_token_key_diagnostics(
    singleton_dictionary: dict[str, object],
    key: str,
    bad: object,
) -> None:
    with pytest.raises(FeatureSchemaError, match=re.escape(f"token 1 {key}")):
        validate_sentence_features([singleton_dictionary | {key: bad}])


@pytest.mark.parametrize(
    "value", [None, [], (), {}, "text", b"text", iter([{}]), [1], [[]], [None]]
)
def test_validator_container_failures(value: object) -> None:
    with pytest.raises(FeatureSchemaError):
        validate_sentence_features(value)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("+1:word.lower", "пес"),
        ("+1:is_title", True),
        ("+1:has_digits", 1),
        ("+1:word.lower", None),
    ],
)
def test_context_disagreement_and_types(key: str, bad: object) -> None:
    value = sentence_features(["Я", "ҐАНОК"])
    value[0][key] = bad  # type: ignore[assignment]
    with pytest.raises(FeatureSchemaError, match="token 1"):
        validate_sentence_features(value)


def test_no_shared_state_or_mutation(singleton_dictionary: dict[str, object]) -> None:
    forms = ["Кіт", "Кіт"]
    original = forms.copy()
    features = sentence_features(forms)
    saved = copy.deepcopy(features)
    restored = validate_sentence_features(
        tuple(MappingProxyType(row) for row in features)
    )
    assert restored == saved
    assert restored is not features and restored[0] is not features[0]
    restored[0]["word.lower"] = "changed"
    assert features == saved
    features[0]["is_title"] = False
    assert features[1]["is_title"] is True
    assert features[1]["-1:is_title"] is True
    forms[0] = "different"
    assert original == ["Кіт", "Кіт"]
    assert sentence_features(original) == saved
    assert sentence_features(["Кіт"]) == [singleton_dictionary]
    assert sentence_features(["ß"])[0]["word.lower"] == "ß"


def test_records_keep_labels_and_provenance_out_of_features(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
) -> None:
    source = artifact_sources[0]
    value = {
        "schema_version": 1,
        "record_type": "gold",
        "sentence_id": sentence_id(
            source.source_role, source.source_path, source.source_sha256, 1
        ),
        "source_path": source.source_path,
        "source_sha256": source.source_sha256,
        "source_role": source.source_role,
        "sentence_ordinal": 1,
        "paragraph_ordinal": None,
        "surface_text": "Кіт",
        "surface_text_origin": "original",
        "token_ids": [1],
        "original_forms": ["Кіт"],
        "normalized_forms": ["Кіт"],
        "token_spans": [None],
        "original_upos": ["ADP"],
        "target_labels": ["ADP"],
        "pseudo": None,
    }
    default = parse_sentence_record(value, config=artifact_config, source=source)
    mapped_config = replace(
        artifact_config,
        tag_mapping=tuple((tag, "FUNC") for tag in ("ADP", "CCONJ", "SCONJ", "PART")),
    )
    mapped = parse_sentence_record(
        value | {"target_labels": ["FUNC"]}, config=mapped_config, source=source
    )
    changed_label = parse_sentence_record(
        value | {"original_upos": ["PROPN"], "target_labels": ["PROPN"]},
        config=artifact_config,
        source=source,
    )
    expected = sentence_features(default.feature_forms)
    for config, record in (
        (artifact_config, default),
        (mapped_config, mapped),
        (artifact_config, changed_label),
    ):
        assert (
            sentence_features(
                record.feature_forms,
                schema_version=config.features.schema_version,
                normalizer_version=config.preprocessing.normalizer_version,
            )
            == expected
        )
        with pytest.raises(FeatureSchemaError):
            sentence_features(record)  # type: ignore[arg-type]
        with pytest.raises(FeatureSchemaError):
            sentence_features(record.to_dict())  # type: ignore[arg-type]
    assert not {"ADP", "FUNC", "PROPN", source.source_path}.intersection(
        expected[0].values()
    )
