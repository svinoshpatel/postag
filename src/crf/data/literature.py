"""Bounded strict raw-text streaming; no labels, corpus fitting or artifact writes."""

import codecs
import hashlib
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from crf.data.identity import (
    InputIdentity,
    iter_literature_identities,
    sentence_id,
    verify_input_identity,
)
from crf.data.normalization import normalize_text
from crf.data.segmentation import (
    AdaptedSentence,
    SegmentationMetadata,
    adapt_sentence,
    segmentation_metadata,
    sentence_ranges,
)
from crf.utils.config import PipelineConfig, ResourceConfig
from crf.utils.resources import MemoryMonitor, check_memory
from crf.utils.serialization import ValidationError, integer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawSentence:
    source: InputIdentity
    sentence_id: str
    paragraph_ordinal: int
    sentence_ordinal: int
    text: AdaptedSentence
    preprocessing: SegmentationMetadata

    @property
    def feature_forms(self) -> tuple[str, ...]:
        return self.text.normalized_forms


@dataclass(frozen=True)
class RawExclusion:
    source: InputIdentity
    reason: Literal["paragraph_chars", "sentence_chars", "sentence_tokens"]
    paragraph_ordinal: int
    sentence_ordinal: int | None
    source_span: tuple[int, int]
    characters: int
    tokens: int | None


@dataclass(frozen=True)
class _Line:
    start: int
    length: int
    text: str | None
    ending: str
    blank: bool


@dataclass(frozen=True)
class _Paragraph:
    start: int
    length: int
    ordinal: int
    text: str | None


def _decoded_chunks(
    source: InputIdentity,
    config: PipelineConfig,
    root: Path,
    monitor: MemoryMonitor | None = None,
) -> Iterator[str]:
    """Validate all consumed bytes, including discarded oversized paragraphs."""
    verify_input_identity(source, config, root, monitor=monitor)
    path = root / source.source_path
    decoder = codecs.getincrementaldecoder(source.encoding)(errors="strict")
    digest = hashlib.sha256()
    consumed = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(config.resources.read_buffer_bytes)
                check_memory(monitor)
                pending = len(decoder.getstate()[0])
                try:
                    decoded = decoder.decode(chunk, final=not chunk)
                except UnicodeError as exc:
                    relative = getattr(exc, "start", 0)
                    raise ValidationError(
                        f"{source.source_path}: {source.encoding} decoding error "
                        f"at byte {consumed - pending + relative}: {exc}"
                    ) from exc
                digest.update(chunk)
                consumed += len(chunk)
                if decoded:
                    yield decoded
                if not chunk:
                    break
    except OSError as exc:
        raise ValidationError(f"{source.source_path}: {exc}") from exc
    if (digest.hexdigest(), consumed) != (source.source_sha256, source.size_bytes):
        raise ValidationError(
            f"{source.source_path}: consumed source checksum/size mismatch"
        )
    verify_input_identity(source, config, root, monitor=monitor)


def _line_units(chunks: Iterable[str]) -> Iterator[str]:
    pending_cr = False
    for chunk in chunks:
        for char in chunk:
            if pending_cr:
                pending_cr = False
                if char == "\n":
                    yield "\r\n"
                    continue
                yield "\r"
            if char == "\r":
                pending_cr = True
            else:
                yield char
    if pending_cr:
        yield "\r"


def _lines(chunks: Iterable[str], limit: int) -> Iterator[_Line]:
    """Count and drain arbitrarily long lines with bounded retained content."""
    start = offset = length = 0
    blank = True
    chars: list[str] = []
    for unit in _line_units(chunks):
        if unit in ("\n", "\r", "\r\n"):
            yield _Line(
                start, length, "".join(chars) if length <= limit else None, unit, blank
            )
            offset += len(unit)
            start, length, blank, chars = offset, 0, True, []
            continue
        if not unit.isspace() and not (offset == 0 and unit == "\ufeff"):
            blank = False
        length += 1
        offset += 1
        if length <= limit:
            chars.append(unit)
        elif chars:
            chars.clear()
    if length:
        yield _Line(
            start, length, "".join(chars) if length <= limit else None, "", blank
        )


def _paragraphs(chunks: Iterable[str], limit: int) -> Iterator[_Paragraph]:
    parts: list[str] = []
    start = length = ordinal = 0
    ending = ""
    active = False
    for line in _lines(chunks, limit):
        if line.blank:
            if active:
                yield _Paragraph(
                    start, length, ordinal, "".join(parts) if length <= limit else None
                )
            parts, length, ending, active = [], 0, "", False
            continue
        if not active:
            start, active = line.start, True
            ordinal += 1
        length += len(ending) + line.length
        if length <= limit and line.text is not None:
            parts.extend((ending, line.text))
        else:
            parts.clear()
        ending = line.ending
    if active:
        yield _Paragraph(
            start, length, ordinal, "".join(parts) if length <= limit else None
        )


def _validate_resources(resources: ResourceConfig) -> None:
    for name in (
        "read_buffer_bytes",
        "max_paragraph_chars",
        "max_sentence_chars",
        "max_sentence_tokens",
        "batch_tokens",
    ):
        integer(getattr(resources, name), name, 1)
    if resources.batch_tokens < resources.max_sentence_tokens:
        raise ValidationError("batch_tokens must be >= max_sentence_tokens")


def iter_literature_events(
    source: InputIdentity,
    *,
    config: PipelineConfig,
    repository_root: Path,
    monitor: MemoryMonitor | None = None,
) -> Iterator[RawSentence | RawExclusion]:
    """Yield sentences/exclusions incrementally; exhaust before accepting a source.

    Oversized paragraphs are drained intact. Gold, overlap, MWT and confidence
    policy are deliberately outside this label-free preprocessing interface.
    """
    if source.source_role != "literature":
        raise ValidationError("Raw literature reader requires a literature source")
    _validate_resources(config.resources)
    check_memory(monitor, expected_budget_mib=config.resources.memory_budget_mib)
    metadata = segmentation_metadata(
        adapter_version=config.preprocessing.tokenizer_adapter_version,
        normalizer_version=config.preprocessing.normalizer_version,
    )
    resources = config.resources
    ordinal = 0
    for paragraph in _paragraphs(
        _decoded_chunks(source, config, repository_root.resolve(), monitor),
        resources.max_paragraph_chars,
    ):
        check_memory(monitor)
        exclusion: RawExclusion | None = None
        if paragraph.text is None:
            exclusion = RawExclusion(
                source,
                "paragraph_chars",
                paragraph.ordinal,
                None,
                (paragraph.start, paragraph.start + paragraph.length),
                paragraph.length,
                None,
            )
            logger.warning(
                "%s: excluded %s (%d characters) at %s",
                source.source_path,
                exclusion.reason,
                exclusion.characters,
                exclusion.source_span,
            )
            yield exclusion
            continue
        normalized = normalize_text(
            paragraph.text,
            source_offset=paragraph.start,
            version=metadata.normalizer_version,
        )
        for start, end in sentence_ranges(normalized):
            check_memory(monitor)
            ordinal += 1
            span = normalized.source_span(start, end)
            characters = span[1] - span[0]
            if characters > resources.max_sentence_chars:
                exclusion = RawExclusion(
                    source,
                    "sentence_chars",
                    paragraph.ordinal,
                    ordinal,
                    span,
                    characters,
                    None,
                )
            else:
                text = adapt_sentence(normalized, start, end)
                check_memory(monitor)
                if not text.normalized_forms:
                    continue
                if len(text.normalized_forms) > resources.max_sentence_tokens:
                    exclusion = RawExclusion(
                        source,
                        "sentence_tokens",
                        paragraph.ordinal,
                        ordinal,
                        span,
                        characters,
                        len(text.normalized_forms),
                    )
                else:
                    yield RawSentence(
                        source,
                        sentence_id(
                            "literature",
                            source.source_path,
                            source.source_sha256,
                            ordinal,
                        ),
                        paragraph.ordinal,
                        ordinal,
                        text,
                        metadata,
                    )
                    continue
            logger.warning(
                "%s: excluded %s (%d characters) at %s",
                source.source_path,
                exclusion.reason,
                exclusion.characters,
                exclusion.source_span,
            )
            yield exclusion


def iter_literature(
    config: PipelineConfig,
    repository_root: Path,
    *,
    monitor: MemoryMonitor | None = None,
) -> Iterator[RawSentence | RawExclusion]:
    """Discover and stream each eligible source separately, without gold reads."""
    for source in iter_literature_identities(config, repository_root, monitor=monitor):
        yield from iter_literature_events(
            source, config=config, repository_root=repository_root, monitor=monitor
        )


def sentence_batches(
    sentences: Iterable[RawSentence],
    *,
    token_limit: int = 4096,
    monitor: MemoryMonitor | None = None,
) -> Iterator[tuple[RawSentence, ...]]:
    """Bound complete-sentence inference batches; never split or silently drop."""
    integer(token_limit, "token_limit", 1)
    check_memory(monitor)
    batch: list[RawSentence] = []
    tokens = 0
    for sentence in sentences:
        check_memory(monitor)
        count = len(sentence.feature_forms)
        if not 0 < count <= token_limit:
            raise ValidationError("Sentence cannot fit in token-bounded batch")
        if tokens + count > token_limit:
            yield tuple(batch)
            batch, tokens = [], 0
        batch.append(sentence)
        tokens += count
    if batch:
        check_memory(monitor)
        yield tuple(batch)
