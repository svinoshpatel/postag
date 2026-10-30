"""Deterministic adapter expectations derived from the raw-text contract."""

import pytest

from crf.data.normalization import normalize_forms
from crf.data.segmentation import iter_adapted_sentences, segmentation_metadata
from crf.utils.serialization import ValidationError


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "північно-західний синьо‐жовтий синьо‑жовтий.",
            (
                "північно",
                "-",
                "західний",
                "синьо",
                "‐",
                "жовтий",
                "синьо",
                "‑",
                "жовтий",
                ".",
            ),
        ),
        (
            "Її сім’я, пʼять, п`ять, п'ять.",
            ("Її", "сім'я", ",", "п'ять", ",", "п'ять", ",", "п'ять", "."),
        ),
        ("Він має 1 000 000.", ("Він", "має", "1 000 000", ".")),
        ("Він має 1\u00a0000\u202f000!", ("Він", "має", "1 000 000", "!")),
        ("Т. Г. Шевченко.", ("Т", ".", "Г", ".", "Шевченко", ".")),
        ("«Кіт» (спить).", ("«", "Кіт", "»", "(", "спить", ")", ".")),
        ("А\u200bБ \u200b Кіт́.", ("А", "\u200b", "Б", "\u200b", "Кіт́", ".")),
        ("\u200b", ("\u200b",)),
        ("\u00ad", ("\u00ad",)),
        (
            "ґанок ҐАНОК г і и ї є ь.",
            ("ґанок", "ҐАНОК", "г", "і", "и", "ї", "є", "ь", "."),
        ),
        ("12,5 01.02.2024 12:30", ("12,5", "01.02.2024", "12:30")),
    ],
)
def test_ud_boundaries_and_preservation(text: str, expected: tuple[str, ...]) -> None:
    result = list(iter_adapted_sentences(text, source_offset=10))
    assert len(result) == 1
    sentence = result[0]
    assert sentence.normalized_forms == expected
    assert normalize_forms(sentence.original_forms) == expected
    previous = 10
    for original, (start, end) in zip(
        sentence.original_forms, sentence.token_spans, strict=True
    ):
        assert previous <= start < end <= len(text) + 10
        assert text[start - 10 : end - 10] == original
        previous = end
    # Every nonwhitespace source code point survives in a token span.
    covered = {
        i for start, end in sentence.token_spans for i in range(start - 10, end - 10)
    }
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_abbreviation_aware_sentence_boundaries() -> None:
    result = list(iter_adapted_sentences("Це проф. Петренко. Він у м. Київ."))
    assert [s.surface_text for s in result] == ["Це проф. Петренко.", "Він у м. Київ."]
    assert result[0].normalized_forms == ("Це", "проф", ".", "Петренко", ".")
    assert result[1].normalized_forms == ("Він", "у", "м", ".", "Київ", ".")


def test_nfc_alignment_and_file_start_bom() -> None:
    text = "\ufeffІ\u0308ї сім’я."
    sentence = list(iter_adapted_sentences(text))[0]
    assert sentence.normalized_forms == ("Її", "сім'я", ".")
    assert sentence.original_forms == ("І\u0308ї", "сім’я", ".")
    assert sentence.token_spans == ((1, 4), (5, 10), (10, 11))
    assert sentence.source_span == (1, 11)
    # A BOM later in a file is preserved as content, not silently removed.
    later = list(iter_adapted_sentences("\ufeffКіт", source_offset=20))[0]
    assert "\ufeff" in "".join(later.original_forms)
    assert "\ufeff" in "".join(later.normalized_forms)


@pytest.mark.parametrize("text", ["", " \t\n", "\ufeff"])
def test_empty_segmentation(text: str) -> None:
    assert list(iter_adapted_sentences(text)) == []


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_bad_adapter_versions(version: object) -> None:
    with pytest.raises(ValidationError, match="adapter version"):
        segmentation_metadata(adapter_version=version)  # type: ignore[arg-type]


def test_metadata_and_bad_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = segmentation_metadata()
    assert metadata.tokenizer_adapter_version == metadata.normalizer_version == 1
    assert metadata.sentence_language == "uk_two"
    assert metadata.unicode_version
    monkeypatch.setattr("crf.data.segmentation.version", lambda _: "999")
    with pytest.raises(ValidationError, match="tokenize-uk version"):
        segmentation_metadata()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("п\u0301’ять", ("п\u0301'ять",)),
        ("1 000 000,5", ("1 000 000,5",)),
        ("1 000 000.25", ("1 000 000.25",)),
        ("а\u0344-б", ("ӓ́", "-", "б")),
        ("сло\u00ad\r\nво.", ("сло\u00ad", "во", ".")),
    ],
)
def test_marked_apostrophes_numeric_groups_and_nfc_units(
    text: str, expected: tuple[str, ...]
) -> None:
    result = list(iter_adapted_sentences(text))
    assert len(result) == 1
    assert result[0].normalized_forms == expected
    assert normalize_forms(result[0].original_forms) == expected


def test_backend_word_omissions_are_recovered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crf.data.segmentation.tokenize_words_with_spans", lambda *args, **kwargs: []
    )
    result = list(iter_adapted_sentences("Кіт \u200b спить."))
    assert result[0].normalized_forms == ("Кіт", "\u200b", "спить", ".")
