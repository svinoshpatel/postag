"""Versioned, case-preserving normalization with conservative source alignment."""

import logging
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)
NORMALIZER_VERSION = 1
_CHARACTERS = str.maketrans(
    {"’": "'", "`": "'", "ʼ": "'", "\u00a0": " ", "\u202f": " "}
)


class NormalizationError(ValueError):
    """Text, version or an alignment request violates the normalization contract."""


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int


@dataclass(frozen=True)
class FormattingCharacter:
    character: str
    source_offset: int
    removed: bool


@dataclass(frozen=True)
class NormalizedText:
    original: str
    text: str
    source_offset: int
    character_spans: tuple[SourceSpan, ...]
    formatting_characters: tuple[FormattingCharacter, ...]
    normalizer_version: int = NORMALIZER_VERSION
    unicode_version: str = unicodedata.unidata_version

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """Project a complete normalized unit range; reject ambiguous partial units."""
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(self.text)
        ):
            raise NormalizationError("expected a nonempty in-bounds normalized span")
        if (
            start > 0 and self.character_spans[start - 1] == self.character_spans[start]
        ) or (
            end < len(self.text)
            and self.character_spans[end - 1] == self.character_spans[end]
        ):
            raise NormalizationError(
                "span boundary splits a Unicode normalization unit"
            )
        return self.character_spans[start].start, self.character_spans[end - 1].end


def _validate(text: str, version: int, *, nonempty: bool = False) -> None:
    if type(version) is not int or version != NORMALIZER_VERSION:
        raise NormalizationError(f"unsupported normalizer version: {version!r}")
    if not isinstance(text, str) or (nonempty and not text):
        raise NormalizationError(
            "expected nonempty token form" if nonempty else "expected text"
        )
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise NormalizationError("text contains invalid Unicode scalar values") from exc


def _formatting(
    text: str, offset: int, *, remove_bom: bool
) -> tuple[FormattingCharacter, ...]:
    observations = tuple(
        FormattingCharacter(
            char, offset + index, remove_bom and index == 0 and char == "\ufeff"
        )
        for index, char in enumerate(text)
        if unicodedata.category(char) == "Cf"
    )
    if observations:
        first = observations[0]
        logger.warning(
            "Unicode formatting characters: %d occurrence(s), %d removed file-start BOM; "
            "first U+%04X at decoded offset %d",
            len(observations),
            sum(item.removed for item in observations),
            ord(first.character),
            first.source_offset,
        )
    return observations


def _line_units(text: str, offset: int) -> Iterator[tuple[str, SourceSpan]]:
    index = int(offset == 0 and text.startswith("\ufeff"))
    while index < len(text):
        char = text[index]
        length = 2 if text.startswith("\r\n", index) else 1
        yield (
            "\n" if char == "\r" else char,
            SourceSpan(offset + index, offset + index + length),
        )
        index += length


def _nfc_units(
    units: Iterator[tuple[str, SourceSpan]],
) -> Iterator[tuple[str, SourceSpan]]:
    """Group nonstarters and interacting starters without prefix-by-prefix NFC."""
    pending: list[str] = []
    start = end = 0
    for char, span in units:
        if not pending:
            pending, start, end = [char], span.start, span.end
            continue
        if unicodedata.combining(unicodedata.normalize("NFD", char)[0]):
            pending.append(char)
            end = span.end
            continue
        normalized = unicodedata.normalize("NFC", "".join(pending))
        next_char = unicodedata.normalize("NFC", char)
        joined = normalized + next_char
        composed = unicodedata.normalize("NFC", joined)
        if composed != joined:
            # Hangul and other zero-CCC compositions cannot be split at this starter.
            pending, end = [composed], span.end
        else:
            yield normalized, SourceSpan(start, end)
            pending, start, end = [char], span.start, span.end
    if pending:
        yield unicodedata.normalize("NFC", "".join(pending)), SourceSpan(start, end)


def normalize_text(
    text: str, *, source_offset: int = 0, version: int = NORMALIZER_VERSION
) -> NormalizedText:
    """Normalize a complete decoded unit, preserving absolute original offsets."""
    _validate(text, version)
    if type(source_offset) is not int or source_offset < 0:
        raise NormalizationError("source_offset must be a nonnegative integer")
    observations = _formatting(text, source_offset, remove_bom=source_offset == 0)
    pieces: list[str] = []
    spans: list[SourceSpan] = []
    for normalized, span in _nfc_units(_line_units(text, source_offset)):
        translated = normalized.translate(_CHARACTERS)
        pieces.append(translated)
        spans.extend([span] * len(translated))
    return NormalizedText(
        text, "".join(pieces), source_offset, tuple(spans), observations
    )


def normalize_form(form: str, *, version: int = NORMALIZER_VERSION) -> str:
    """Normalize one token without removing a BOM or changing token boundaries."""
    _validate(form, version, nonempty=True)
    _formatting(form, 0, remove_bom=False)
    return unicodedata.normalize(
        "NFC", form.replace("\r\n", "\n").replace("\r", "\n")
    ).translate(_CHARACTERS)


def normalize_forms(
    forms: Sequence[str], *, version: int = NORMALIZER_VERSION
) -> tuple[str, ...]:
    """Normalize each form independently; never merge adjacent gold words."""
    if isinstance(forms, (str, bytes)) or not isinstance(forms, Sequence) or not forms:
        raise NormalizationError("expected a nonempty sequence of token forms")
    return tuple(normalize_form(form, version=version) for form in forms)


def lexical_form(normalized_form: str, *, version: int = NORMALIZER_VERSION) -> str:
    """Lowercase an already normalized form and apply NFC again; never casefold."""
    _validate(normalized_form, version, nonempty=True)
    return unicodedata.normalize("NFC", normalized_form.lower())
