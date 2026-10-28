"""Backend checks; full UD adaptation and file streaming belong to milestone 4."""

from dataclasses import FrozenInstanceError

import pytest

from crf.data.tokenization import tokenize_paragraph


def test_abbreviations_and_line_breaks() -> None:
    result = tokenize_paragraph("Це проф. Петренко. Він у м. Київ.")
    assert [item.sentence.text for item in result] == [
        "Це проф. Петренко.",
        "Він у м. Київ.",
    ]
    assert len(tokenize_paragraph("Перший\nрядок без крапки")) == 1
    assert len(tokenize_paragraph("Т. Г. Шевченко написав вірш.")) == 1


def test_original_slices_and_immutable_records() -> None:
    text = "  Її сім’я, пʼять, п'ять. «Київ» — Україна!"
    result = tokenize_paragraph(text)
    assert len(result) == 2
    for item in result:
        assert text[item.sentence.start : item.sentence.end] == item.sentence.text
        previous_end = item.sentence.start
        for token in item.tokens:
            assert previous_end <= token.start < token.end <= item.sentence.end
            assert text[token.start : token.end] == token.text
            previous_end = token.end
    assert "сім’я" in [token.text for item in result for token in item.tokens]
    assert "пʼять" in [token.text for item in result for token in item.tokens]
    with pytest.raises(FrozenInstanceError):
        result[0].sentence.start = 0  # type: ignore[misc]


@pytest.mark.parametrize("text", ["", " \n\t", "\r\n"])
def test_empty_text(text: str) -> None:
    assert tokenize_paragraph(text) == ()


def test_final_fragment_and_punctuation() -> None:
    assert tokenize_paragraph("Без крапки")[0].sentence.text == "Без крапки"
    assert [span.text for span in tokenize_paragraph("Привіт, Київ!")[0].tokens] == [
        "Привіт",
        ",",
        "Київ",
        "!",
    ]
