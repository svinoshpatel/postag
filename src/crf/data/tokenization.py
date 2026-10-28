"""Low-level span backend; production normalization/UD adaptation comes later."""

from dataclasses import dataclass

from tokenize_uk import tokenize_sents_with_spans, tokenize_words_with_spans


@dataclass(frozen=True)
class TextSpan:
    """Original text and end-exclusive code-point offsets into the input."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenizedSentence:
    """Immutable sentence and token spans relative to the supplied paragraph."""

    sentence: TextSpan
    tokens: tuple[TextSpan, ...]


def tokenize_paragraph(text: str) -> tuple[TokenizedSentence, ...]:
    """Segment one supplied paragraph using non-legacy Ukrainian span APIs.

    Recover text from source offsets because upstream token strings may normalize
    apostrophes. This backend does not claim UD-compatible token boundaries.
    """
    result: list[TokenizedSentence] = []
    for _, start, end in tokenize_sents_with_spans(
        text, legacy=False, language_code="uk_two"
    ):
        sentence = text[start:end]
        tokens = tuple(
            TextSpan(sentence[left:right], start + left, start + right)
            for _, left, right in tokenize_words_with_spans(sentence, legacy=False)
        )
        if tokens:
            result.append(TokenizedSentence(TextSpan(sentence, start, end), tokens))
    return tuple(result)
