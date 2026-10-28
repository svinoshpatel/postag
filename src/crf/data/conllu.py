"""Strict sentence-streaming gold parser and independent text-only audit reader."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from conllu.exceptions import ParseException
from conllu.parser import (
    parse_comment_line,
    parse_id_value,
)

from crf.data.identity import (
    GOLD_ROLES,
    InputIdentity,
    sentence_id,
    verify_input_identity,
)
from crf.data.labels import TagMappingError, validate_upos
from crf.data.normalization import NormalizationError, normalize_forms
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.utils.config import ConfigError, PipelineConfig
from crf.utils.serialization import ValidationError

type RowID = int | tuple[int, Literal["-", "."], int]


@dataclass(frozen=True)
class SurfaceRow:
    """Only reconstruction fields; never any annotation-column content."""

    token_id: RowID
    form: str
    spacing: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GoldAuditSentence:
    sentence_id: str
    source: InputIdentity
    sentence_ordinal: int
    line_number: int
    token_ids: tuple[int, ...]
    original_forms: tuple[str, ...]
    surface_text: str
    surface_text_origin: Literal["original", "reconstructed"]
    sent_id: str | None
    document_id: str | None
    paragraph_id: str | None
    surface_rows: tuple[SurfaceRow, ...]


@dataclass(frozen=True)
class GoldSentence:
    """Pre-normalization parser output; feature extraction cannot consume it."""

    audit: GoldAuditSentence
    original_upos: tuple[str, ...]
    target_labels: tuple[str, ...]
    comments: tuple[str, ...]

    def to_record(
        self, config: PipelineConfig, *, final_evaluation: bool = False
    ) -> SentenceRecord:
        """Normalize a parsed sentence without publishing or guessing word offsets."""
        audit = self.audit
        if type(final_evaluation) is not bool:
            raise ValidationError("final_evaluation: expected boolean")
        if audit.source.source_role == "test_gold" and not final_evaluation:
            raise ValidationError(
                "test_gold records are restricted to final evaluation"
            )
        try:
            normalized = normalize_forms(
                audit.original_forms, version=config.preprocessing.normalizer_version
            )
        except NormalizationError as exc:
            raise _error(
                Path(audit.source.source_path), audit.line_number, str(exc)
            ) from exc
        record = SentenceRecord(
            schema_version=1,
            record_type="gold",
            sentence_id=audit.sentence_id,
            source_path=audit.source.source_path,
            source_sha256=audit.source.source_sha256,
            source_role=audit.source.source_role,
            sentence_ordinal=audit.sentence_ordinal,
            paragraph_ordinal=None,
            surface_text=audit.surface_text,
            surface_text_origin=audit.surface_text_origin,
            token_ids=audit.token_ids,
            original_forms=audit.original_forms,
            normalized_forms=normalized,
            token_spans=(None,) * len(audit.token_ids),
            original_upos=self.original_upos,
            target_labels=self.target_labels,
            pseudo=None,
        )
        return parse_sentence_record(
            record.to_dict(),
            config=config,
            source=audit.source,
            final_evaluation=final_evaluation,
        )


def _error(path: Path, line: int, reason: str) -> ValidationError:
    return ValidationError(f"{path}:{line}: {reason}")


def _blocks(path: Path, source: InputIdentity) -> Iterator[tuple[tuple[int, str], ...]]:
    """Keep at most one sentence; decode physical lines for precise diagnostics."""
    block: list[tuple[int, str]] = []
    digest = hashlib.sha256()
    size = 0
    line_number = 0
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                size += len(raw)
                try:
                    line = raw.decode("utf-8", errors="strict").removesuffix("\n")
                except UnicodeDecodeError as exc:
                    raise _error(path, line_number, "invalid UTF-8") from exc
                line = line.removesuffix("\r")
                if "\r" in line or (line_number == 1 and line.startswith("\ufeff")):
                    raise _error(path, line_number, "unexpected CR or BOM")
                if not line.strip():
                    if not raw.endswith(b"\n"):
                        raise _error(path, line_number, "unterminated blank delimiter")
                    if block:
                        yield tuple(block)
                        block.clear()
                else:
                    block.append((line_number, line))
    except OSError as exc:
        raise _error(path, line_number or 1, str(exc)) from exc
    if block:
        raise _error(path, line_number, "missing final blank sentence delimiter")
    if (digest.hexdigest(), size) != (source.source_sha256, source.size_bytes):
        raise _error(path, line_number or 1, "source changed while parsing")


def _row_id(value: str, path: Path, line: int) -> RowID:
    try:
        parsed = cast(RowID | None, parse_id_value(value))
    except (ParseException, ValueError) as exc:
        raise _error(path, line, "invalid CoNLL-U ID") from exc
    if parsed is None or (isinstance(parsed, int) and parsed < 1):
        raise _error(path, line, "word ID must be a positive integer")
    if isinstance(parsed, tuple):
        start, kind, end = parsed
        if (kind == "-" and start >= end) or value != f"{start}{kind}{end}":
            raise _error(path, line, "invalid range/empty-node ID")
    return parsed


def _spacing(value: str, path: Path, line: int) -> tuple[tuple[str, str], ...]:
    result: dict[str, str] = {}
    for entry in value.split("|"):
        key, separator, item = entry.partition("=")
        if key not in {"SpaceAfter", "SpacesAfter", "SpacesBefore"}:
            continue
        if not separator or not item or key in result:
            raise _error(path, line, "invalid/duplicate spacing metadata")
        if key == "SpaceAfter" and item != "No":
            raise _error(path, line, "SpaceAfter must be No")
        if key != "SpaceAfter":
            _unescape(item, path, line)
        result[key] = item
    return tuple(result.items())


def _unescape(value: str, path: Path, line: int) -> str:
    escapes = {"s": " ", "t": "\t", "n": "\n", "r": "\r", "p": "|", "\\": "\\"}
    output: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\":
            index += 1
            if index == len(value) or value[index] not in escapes:
                raise _error(path, line, "invalid spacing escape")
            char = escapes[value[index]]
        output.append(char)
        index += 1
    return "".join(output)


def _reconstruct(rows: tuple[SurfaceRow, ...], path: Path, line: int) -> str:
    surface: list[SurfaceRow] = []
    covered = 0
    for row in rows:
        if isinstance(row.token_id, tuple):
            if row.token_id[1] == ".":
                continue
            covered = row.token_id[2]
        elif row.token_id <= covered:
            continue
        surface.append(row)
    parts: list[str] = []
    for index, row in enumerate(surface):
        spacing = dict(row.spacing)
        parts.append(_unescape(spacing.get("SpacesBefore", ""), path, line))
        parts.append(row.form)
        if "SpacesAfter" in spacing:
            parts.append(_unescape(spacing["SpacesAfter"], path, line))
        elif spacing.get("SpaceAfter") != "No" and index + 1 < len(surface):
            # Explicit spacing on the next surface row supersedes the default gap.
            if "SpacesBefore" not in dict(surface[index + 1].spacing):
                parts.append(" ")
    return "".join(parts)


def _parse_block(
    block: tuple[tuple[int, str], ...],
    *,
    path: Path,
    source: InputIdentity,
    ordinal: int,
    document_id: str | None,
    paragraph_id: str | None,
    labeled: bool,
) -> tuple[GoldAuditSentence, tuple[str, ...], tuple[str, ...]]:
    rows: list[SurfaceRow] = []
    ids: list[int] = []
    forms: list[str] = []
    labels: list[str] = []
    comments: list[str] = []
    seen: set[RowID] = set()
    metadata: dict[str, str] = {}
    range_end = 0
    for number, line in block:
        if line.startswith("#"):
            if rows:
                raise _error(path, number, "comments must precede annotation rows")
            if labeled:
                comments.append(line)
            # Whitelist before parsing: arbitrary comments never enter audit metadata.
            key = line[1:].partition("=")[0].strip()
            if key in {"sent_id", "text", "newdoc id", "newpar id"}:
                pairs = parse_comment_line(line)
                if key in metadata:
                    raise _error(path, number, f"duplicate {key} comment")
                if not pairs or not pairs[0][1]:
                    raise _error(path, number, f"empty {key} comment")
                metadata[key] = pairs[0][1]
            elif key in {"newdoc", "newpar"}:
                # Anonymous boundaries cannot establish an unambiguous document ID.
                metadata[key + " id"] = ""
            continue
        fields = line.split("\t")
        if len(fields) != 10 or any(not field for field in fields):
            raise _error(path, number, "expected ten nonempty tab-separated fields")
        identifier = _row_id(fields[0], path, number)
        if identifier in seen:
            raise _error(path, number, "duplicate token ID")
        seen.add(identifier)
        if isinstance(identifier, int):
            if identifier != len(ids) + 1:
                raise _error(path, number, "word IDs must be consecutive from 1")
            ids.append(identifier)
            forms.append(fields[1])
            if labeled:
                try:
                    labels.append(validate_upos(fields[3]))
                except TagMappingError as exc:
                    raise _error(path, number, str(exc)) from exc
        elif identifier[1] == "-":
            start, _, end = identifier
            if start != len(ids) + 1 or start <= range_end:
                raise _error(path, number, "misplaced or overlapping multiword range")
            range_end = end
        elif identifier[0] != len(ids):
            raise _error(path, number, "empty node must follow its word anchor")
        rows.append(
            SurfaceRow(identifier, fields[1], _spacing(fields[9], path, number))
        )
    if not ids:
        raise _error(path, block[0][0], "sentence has no integer-ID words")
    if range_end > len(ids):
        raise _error(path, block[-1][0], "multiword range exceeds sentence words")
    if "newdoc id" in metadata:
        document_id = metadata["newdoc id"] or None
        paragraph_id = None
    if "newpar id" in metadata:
        paragraph_id = metadata["newpar id"] or None
    frozen_rows = tuple(rows)
    surface = metadata.get("text")
    audit = GoldAuditSentence(
        sentence_id(
            source.source_role, source.source_path, source.source_sha256, ordinal
        ),
        source,
        ordinal,
        block[0][0],
        tuple(ids),
        tuple(forms),
        surface
        if surface is not None
        else _reconstruct(frozen_rows, path, block[0][0]),
        "original" if surface is not None else "reconstructed",
        metadata.get("sent_id"),
        document_id,
        paragraph_id,
        frozen_rows,
    )
    return audit, tuple(labels), tuple(comments)


def _iterate(
    source: InputIdentity,
    config: PipelineConfig,
    repository_root: Path,
    *,
    labeled: bool,
) -> Iterator[tuple[GoldAuditSentence, tuple[str, ...], tuple[str, ...]]]:
    if source.source_role not in GOLD_ROLES:
        raise ValidationError("CoNLL-U readers require a gold source role")
    verify_input_identity(source, config, repository_root)
    path = repository_root.resolve() / source.source_path
    document_id = paragraph_id = None
    for ordinal, block in enumerate(_blocks(path, source), 1):
        result = _parse_block(
            block,
            path=path,
            source=source,
            ordinal=ordinal,
            document_id=document_id,
            paragraph_id=paragraph_id,
            labeled=labeled,
        )
        document_id, paragraph_id = result[0].document_id, result[0].paragraph_id
        yield result


def iter_gold_audit(
    source: InputIdentity, config: PipelineConfig, repository_root: Path
) -> Iterator[GoldAuditSentence]:
    """Read only surface fields, including test text without annotation interpretation."""
    for audit, _, _ in _iterate(source, config, repository_root, labeled=False):
        yield audit


def iter_gold_sentences(
    source: InputIdentity,
    config: PipelineConfig,
    repository_root: Path,
    *,
    final_evaluation: bool = False,
) -> Iterator[GoldSentence]:
    """Load original gold labels; test labels require an explicit evaluation opt-in."""
    if type(final_evaluation) is not bool:
        raise ValidationError("final_evaluation: expected boolean")
    if source.source_role == "test_gold" and not final_evaluation:
        raise ValidationError("test_gold labels are restricted to final evaluation")
    try:
        label_space = config.label_space
    except ConfigError as exc:
        raise ValidationError(str(exc)) from exc
    for audit, labels, comments in _iterate(
        source, config, repository_root, labeled=True
    ):
        targets = label_space.map_upos(labels)
        yield GoldSentence(audit, labels, targets, comments)


def iter_gold_records(
    source: InputIdentity,
    config: PipelineConfig,
    repository_root: Path,
    *,
    final_evaluation: bool = False,
) -> Iterator[SentenceRecord]:
    """Stream validated normalized gold records; retain parser metadata separately."""
    for parsed in iter_gold_sentences(
        source, config, repository_root, final_evaluation=final_evaluation
    ):
        yield parsed.to_record(config, final_evaluation=final_evaluation)
