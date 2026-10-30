"""Versioned, label-free Ukrainian UD boundary adapter with source alignment."""

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version

from tokenize_uk import tokenize_sents_with_spans, tokenize_words_with_spans

from crf.data.normalization import NormalizedText, normalize_text
from crf.utils.serialization import ValidationError

TOKENIZER_ADAPTER_VERSION = 1
TOKENIZER_PACKAGE_VERSION = "2.0.0"
SENTENCE_LANGUAGE = "uk_two"
_DIGIT_GROUP = e.compile(r"(?<![\w.,])[0-9]{1,3}(?: [0-9]{3})+(?:[.,][0-9]+)?(?!\w)")


@dataclass(frozen=True)
class AdaptedSentence:
    """Original surface and forms with absolute end-exclusive decoded offsets."""

    surface_text: str
    source_span: tuple[int, int]
    original_forms: tuple[str, ...]
    normalized_forms: tuple[str, ...]
    token_spans: tuple[tuple[int, int], ...]

    @property
    def feature_forms(self) -> tuple[str, ...]:
        return self.normalized_forms


@dataclass(frozen=True)
class SegmentationMetadata:
    tokenizer_adapter_version: int
    tokenizer_package_version: str
    sentence_language: str
    normalizer_version: int
    unicode_version: str


def segmentation_metadata(
    *, adapter_version: int = 1, normalizer_version: int = 1
) -> SegmentationMetadata:
    """Validate recorded preprocessing versions, including the installed backend."""
    if type(adapter_version) is not int or adapter_version != TOKENIZER_ADAPTER_VERSION:
        raise ValidationError(
            f"Unsupported tokenizer adapter version: {adapter_version!r}"
        )
    normalized = normalize_text("", version=normalizer_version)
    installed = version("tokenize-uk")
    if installed != TOKENIZER_PACKAGE_VERSION:
        raise ValidationError(f"Unsupported tokenize-uk version: {installed}")
    return SegmentationMetadata(
        adapter_version,
        installed,
        SENTENCE_LANGUAGE,
        normalizer_version,
        normalized.unicode_version,
    )


def _unit_boundary(text: NormalizedText, index: int) -> bool:
    return index in (0, len(text.text)) or (
        text.character_spans[index - 1] != text.character_spans[index]
    )


def sentence_ranges(text: NormalizedText) -> Iterator[tuple[int, int]]:
    """Segment case-preserving normalized text; preserve uncovered content."""
    cursor = 0
    for _, start, end in tokenize_sents_with_spans(
        text.text, legacy=False, language_code=SENTENCE_LANGUAGE
    ):
        if not cursor <= start < end <= len(text.text):
            raise ValidationError("Invalid backend sentence spans")
        if text.text[cursor:start].strip():
            yield cursor, start
        yield start, end
        cursor = end
    if text.text[cursor:].strip():
        yield cursor, len(text.text)


def _token_ranges(text: str) -> Iterator[tuple[int, int]]:
    """Adapt backend cuts, adding omitted content without backend string rewrites."""
    cuts = {0, len(text)}
    previous = 0
    for _, start, end in tokenize_words_with_spans(text, legacy=False):
        if not previous <= start < end <= len(text):
            raise ValidationError("Invalid backend token spans")
        cuts.update((start, end))
        previous = end
    protected: set[int] = set()
    for index, char in enumerate(text):
        previous_letter = index - 1
        while (
            char == "'"
            and previous_letter >= 0
            and unicodedata.category(text[previous_letter]).startswith("M")
        ):
            previous_letter -= 1
        internal_apostrophe = (
            char == "'"
            and 0 < index < len(text) - 1
            and previous_letter >= 0
            and text[previous_letter].isalpha()
            and text[index + 1].isalpha()
        )
        numeric_separator = (
            char in ".,:"
            and 0 < index < len(text) - 1
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        )
        if internal_apostrophe or numeric_separator:
            protected.update((index, index + 1))
        if char.isspace() or (
            (unicodedata.category(char).startswith("P") or char in "-‐‑")
            and not internal_apostrophe
            and not numeric_separator
        ):
            cuts.update((index, index + 1))
    cuts.difference_update(protected)
    # Backend numeric grouping can stop early before sentence punctuation.
    for match in _DIGIT_GROUP.finditer(text):
        cuts.difference_update(range(match.start() + 1, match.end()))
        cuts.update((match.start(), match.end()))
    ordered = sorted(cuts)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if text[left:right].strip():
            yield left, right


def adapt_sentence(text: NormalizedText, start: int, end: int) -> AdaptedSentence:
    """Project adapted boundaries to original text without splitting NFC units."""
    source_start, source_end = text.source_span(start, end)
    cuts = {start, end}
    for left, right in _token_ranges(text.text[start:end]):
        cuts.update((start + left, start + right))
    # Joining cuts within an NFC unit preserves its entire original representation.
    ordered = sorted(cut for cut in cuts if _unit_boundary(text, cut))
    forms: list[str] = []
    originals: list[str] = []
    spans: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        form = text.text[left:right]
        if not form.strip():
            continue
        span = text.source_span(left, right)
        forms.append(form)
        spans.append(span)
        originals.append(
            text.original[span[0] - text.source_offset : span[1] - text.source_offset]
        )
    return AdaptedSentence(
        text.original[
            source_start - text.source_offset : source_end - text.source_offset
        ],
        (source_start, source_end),
        tuple(originals),
        tuple(forms),
        tuple(spans),
    )


def iter_adapted_sentences(
    paragraph: str,
    *,
    source_offset: int = 0,
    adapter_version: int = 1,
    normalizer_version: int = 1,
) -> Iterator[AdaptedSentence]:
    """Reusable complete-paragraph adapter for literature and standalone inference.

    Callers bound paragraphs and apply sentence/resource policy. This API does not
    apply contamination, confidence or deduplication filters.
    """
    segmentation_metadata(
        adapter_version=adapter_version, normalizer_version=normalizer_version
    )
    normalized = normalize_text(
        paragraph, source_offset=source_offset, version=normalizer_version
    )
    for start, end in sentence_ranges(normalized):
        result = adapt_sentence(normalized, start, end)
        if result.normalized_forms:
            yield result
