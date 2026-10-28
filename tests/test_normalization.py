"""Offline normalization and original-code-point alignment fixtures."""

import logging
import random
import unicodedata
from dataclasses import FrozenInstanceError, replace
from itertools import pairwise
from pathlib import Path

import pytest

from crf.data.conllu import iter_gold_audit, iter_gold_records, iter_gold_sentences
from crf.data.identity import InputIdentity, SourceRole, digest_file
from crf.data.normalization import (
    NORMALIZER_VERSION,
    NormalizationError,
    SourceSpan,
    lexical_form,
    normalize_form,
    normalize_forms,
    normalize_text,
)
from crf.data.records import parse_sentence_record
from crf.features.surface import normalize_form as old_normalize_form
from crf.features.surface import sentence_features
from crf.utils.config import PipelineConfig
from crf.utils.serialization import ValidationError, jsonl_bytes, loads_json


@pytest.mark.parametrize("apostrophe", ["'", "’", "`", "ʼ"])
def test_apostrophes_and_case_preservation(apostrophe: str) -> None:
    original = f"П{apostrophe}ЯТЬ"
    assert normalize_text(original).text == "П'ЯТЬ"
    assert normalize_form(original) == "П'ЯТЬ"
    assert lexical_form(normalize_form(original)) == "п'ять"
    assert original == f"П{apostrophe}ЯТЬ"
    assert old_normalize_form is normalize_form


def test_original_offsets_after_bom_composition_and_newlines() -> None:
    original = "\ufeffі\u0308\r\nп’ять\u00a0Ґ\rX"
    result = normalize_text(original)
    assert result.original == original
    assert result.text == "ї\nп'ять Ґ\nX"
    assert result.character_spans == tuple(
        SourceSpan(*pair)
        for pair in (
            (1, 3),
            (3, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
        )
    )
    assert result.source_span(0, 1) == (1, 3)
    assert result.source_span(1, 2) == (3, 5)
    assert result.source_span(2, 7) == (5, 10)
    assert result.source_span(0, len(result.text)) == (1, len(original))
    assert result.normalizer_version == NORMALIZER_VERSION == 1
    assert result.unicode_version == unicodedata.unidata_version
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.character_spans[0].start = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("і\u0308", "ї"),
        ("a\u0315\u0300", "à\u0315"),
        ("а\u0301", "а\u0301"),
        ("\u0344", "\u0308\u0301"),
        ("\u212b", "Å"),
        ("\u1100\u1161\u11a8", "각"),
        ("\uac00\u11a8", "각"),
        ("\u09c7\u09be", "ো"),
        ("\u0bc6\u0bbe", "ொ"),
        ("\u0301\u0323", "\u0323\u0301"),
    ],
)
def test_complete_unicode_units_and_conservative_spans(
    original: str, expected: str
) -> None:
    result = normalize_text(original, source_offset=123)
    assert result.text == expected
    assert result.character_spans == (SourceSpan(123, 123 + len(original)),) * len(
        expected
    )
    assert result.source_span(0, len(expected)) == (123, 123 + len(original))
    for cut in range(1, len(expected)):
        with pytest.raises(NormalizationError, match="normalization unit"):
            result.source_span(0, cut)
        with pytest.raises(NormalizationError, match="normalization unit"):
            result.source_span(cut, len(expected))


def test_absolute_offsets_are_codepoints_and_allow_separate_paragraphs() -> None:
    original = "😀\r\n\r\nі\u0308’"
    result = normalize_text(original[5:], source_offset=5)
    assert result.text == "ї'"
    assert result.source_span(0, 1) == (5, 7)
    assert result.source_span(1, 2) == (7, 8)
    assert normalize_text("😀").source_span(0, 1) == (0, 1)


@pytest.mark.parametrize(
    ("original", "offset", "expected"),
    [
        ("\ufeffКіт", 0, "Кіт"),
        ("\ufeffКіт", 9, "\ufeffКіт"),
        (" \ufeffКіт", 0, " \ufeffКіт"),
        ("\ufeff\ufeffКіт", 0, "\ufeffКіт"),
        ("Кіт\ufeff", 0, "Кіт\ufeff"),
    ],
)
def test_bom_removed_only_once_at_file_start(
    original: str, offset: int, expected: str
) -> None:
    result = normalize_text(original, source_offset=offset)
    assert result.text == expected
    assert sum(item.removed for item in result.formatting_characters) == int(
        offset == 0 and original.startswith("\ufeff")
    )
    assert normalize_form(original) == original


@pytest.mark.parametrize("original", ["", "\ufeff"])
def test_empty_normalized_text_has_no_projectable_span(original: str) -> None:
    result = normalize_text(original)
    assert result.text == ""
    assert result.character_spans == ()
    with pytest.raises(NormalizationError):
        result.source_span(0, 0)


def test_whitespace_paragraphs_cyrillic_lookalikes_and_stress() -> None:
    original = " ҐгГгИиІіЕеЄєЇїьЬ AА IІ eе\u0301\u00a0\u202f\t\r\n\r\nкіт\r "
    expected = " ҐгГгИиІіЕеЄєЇїьЬ AА IІ eе\u0301  \t\n\nкіт\n "
    assert normalize_text(original).text == expected
    assert normalize_form(original) == expected
    assert "AА IІ eе" in expected
    assert normalize_text("  а\t\tб\n\nв  ").text == "  а\t\tб\n\nв  "


def test_formatting_observations_logging_and_preservation(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = "\ufeffа\u00adб\u200b\u200c\u200d\u2060\ufeff"
    with caplog.at_level(logging.WARNING, logger="crf.data.normalization"):
        result = normalize_text(original)
    assert result.text == original[1:]
    assert [
        (item.character, item.source_offset, item.removed)
        for item in result.formatting_characters
    ] == [
        ("\ufeff", 0, True),
        ("\u00ad", 2, False),
        ("\u200b", 4, False),
        ("\u200c", 5, False),
        ("\u200d", 6, False),
        ("\u2060", 7, False),
        ("\ufeff", 8, False),
    ]
    assert "7 occurrence(s), 1 removed" in caplog.text
    assert "decoded offset 0" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="crf.data.normalization"):
        assert normalize_form(original) == original
    assert "7 occurrence(s), 0 removed" in caplog.text
    assert capsys.readouterr().out == ""


def test_form_sequences_and_lexical_casing() -> None:
    original = ["і", "\u0308", "ҐАНОК", "І\u0308", "ß", "İ"]
    normalized = normalize_forms(original)
    assert normalized == ("і", "\u0308", "ҐАНОК", "Ї", "ß", "İ")
    assert original == ["і", "\u0308", "ҐАНОК", "І\u0308", "ß", "İ"]
    assert lexical_form("ҐАНОК") == "ґанок"
    assert lexical_form("ß") == "ß"
    assert lexical_form("İ") == "i\u0307"
    features = sentence_features(["ҐАНОК", "І\u0308"])
    assert features[0]["is_uppercase"] is True
    assert features[0]["word.lower"] == "ґанок"
    assert features[1]["word.lower"] == "ї"
    assert features[1]["is_title"] is True


@pytest.mark.parametrize("text", [None, b"text", 1, True, [], "\ud800", "x\udfff"])
def test_invalid_text(text: object) -> None:
    for operation in (normalize_text, normalize_form, lexical_form):
        with pytest.raises(NormalizationError):
            operation(text)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", [0, 2, True, False, "1", None])
def test_unsupported_versions(version: object) -> None:
    for operation in (normalize_text, normalize_form, lexical_form):
        with pytest.raises(NormalizationError, match="version"):
            operation("Кіт", version=version)  # type: ignore[arg-type]
    with pytest.raises(NormalizationError, match="version"):
        normalize_forms(["Кіт"], version=version)  # type: ignore[arg-type]


@pytest.mark.parametrize("offset", [-1, 1.5, True, "0", None])
def test_invalid_source_offsets(offset: object) -> None:
    with pytest.raises(NormalizationError, match="source_offset"):
        normalize_text("Кіт", source_offset=offset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 1), (0, 4), (1, 1), (2, 1), (True, 2), (0, False), (0.0, 1), (0, "2")],
)
def test_invalid_projection_indices(start: object, end: object) -> None:
    with pytest.raises(NormalizationError, match="in-bounds"):
        normalize_text("Кіт").source_span(start, end)  # type: ignore[arg-type]


@pytest.mark.parametrize("forms", [[], (), "Кіт", b"word", None, [""], ["Кіт", None]])
def test_invalid_form_sequences(forms: object) -> None:
    with pytest.raises(NormalizationError):
        normalize_forms(forms)  # type: ignore[arg-type]
    with pytest.raises(NormalizationError):
        normalize_form("")


def test_bounded_unicode_reference_and_all_complete_unit_projections() -> None:
    # Fixed seed and small text units exercise interactions, not production corpora.
    generator = random.Random(42)
    alphabet = (
        "і",
        "\u0308",
        "a",
        "\u0315",
        "\u0300",
        "\u0323",
        "\u0344",
        "\u212b",
        "\u1100",
        "\u1161",
        "\u11a8",
        "\u09c7",
        "\u09be",
        "’",
        "\u00a0",
        "\r",
        "\n",
        "Ґ",
    )
    for _ in range(80):
        original = "".join(generator.choices(alphabet, k=16))
        expected = (
            unicodedata.normalize(
                "NFC", original.replace("\r\n", "\n").replace("\r", "\n")
            )
            .replace("’", "'")
            .replace("\u00a0", " ")
        )
        result = normalize_text(original, source_offset=20)
        assert result.text == expected
        boundaries = (
            [0]
            + [
                index
                for index in range(1, len(result.text))
                if result.character_spans[index - 1] != result.character_spans[index]
            ]
            + [len(result.text)]
        )
        for left, right in pairwise(boundaries):
            start, end = result.source_span(left, right)
            assert 20 <= start < end <= 20 + len(original)
            assert (
                normalize_form(original[start - 20 : end - 20])
                == result.text[left:right]
            )
        for left, middle, right in zip(
            boundaries[:-2], boundaries[1:-1], boundaries[2:], strict=True
        ):
            assert (
                result.source_span(left, middle)[1]
                <= result.source_span(middle, right)[0]
            )


@pytest.mark.parametrize("role", ["train_gold", "val_gold", "test_gold"])
def test_gold_conversion_preserves_originals_mapping_and_null_spans(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    role: SourceRole,
) -> None:
    config = replace(
        artifact_config,
        tag_mapping=tuple((tag, "FUNC") for tag in ("ADP", "CCONJ", "SCONJ", "PART")),
    )
    path = artifact_repository / getattr(config.data, role)
    content = "# sent_id = fixture\n# text = Ї сім’я\n1\tІ\u0308\t_\tPART\t_\t_\t0\troot\t_\t_\n2\tсім’я\t_\tNOUN\t_\t_\t1\tdep\t_\t_\n\n"
    path.write_text(content, encoding="utf-8")
    digest = digest_file(path)
    source = InputIdentity(
        1,
        role,
        getattr(config.data, role).as_posix(),
        digest.sha256,
        digest.size_bytes,
        "utf-8",
    )
    final = role == "test_gold"
    (parsed,) = iter_gold_sentences(
        source, config, artifact_repository, final_evaluation=final
    )
    record = parsed.to_record(config, final_evaluation=final)
    assert record.original_forms == ("І\u0308", "сім’я")
    assert record.normalized_forms == ("Ї", "сім'я")
    assert record.original_upos == ("PART", "NOUN")
    assert record.target_labels == ("FUNC", "NOUN")
    assert record.token_spans == (None, None)
    assert record.surface_text == "Ї сім’я"
    assert record.sentence_id == parsed.audit.sentence_id
    assert record.feature_forms == ("Ї", "сім'я")
    assert parsed.comments == ("# sent_id = fixture", "# text = Ї сім’я")
    assert list(iter_gold_audit(source, config, artifact_repository)) == [parsed.audit]
    assert list(
        iter_gold_records(source, config, artifact_repository, final_evaluation=final)
    ) == [record]
    restored = parse_sentence_record(
        loads_json(jsonl_bytes(record.to_dict())),
        config=config,
        source=source,
        final_evaluation=final,
    )
    assert restored == record
    assert path.read_bytes() == content.encode("utf-8")
    if final:
        with pytest.raises(ValidationError, match="final evaluation"):
            parsed.to_record(config)
        with pytest.raises(ValidationError, match="final evaluation"):
            list(iter_gold_records(source, config, artifact_repository))
    with pytest.raises(ValidationError, match="mapped original_upos"):
        parsed.to_record(artifact_config, final_evaluation=final)
    unsupported = replace(
        config, preprocessing=replace(config.preprocessing, normalizer_version=2)
    )
    with pytest.raises(ValidationError, match=r"conllu:1:.*version"):
        parsed.to_record(unsupported, final_evaluation=final)
    with pytest.raises(ValidationError, match="boolean"):
        parsed.to_record(config, final_evaluation=1)  # type: ignore[arg-type]
