"""Text-only, disk-backed pseudo-label contamination controls.

Mutable audit state is necessary for incremental SQLite indexing, counters and
single-use completion. Returned decisions/reports are immutable. Original sources
are read only; standalone inference intentionally bypasses this policy.
"""

import hashlib
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from crf.data.conllu import iter_gold_audit
from crf.data.identity import GOLD_ROLES, InputIdentity, iter_input_identities
from crf.data.literature import RawExclusion, RawSentence, iter_literature_events
from crf.data.normalization import normalize_form
from crf.data.segmentation import (
    SegmentationMetadata,
    iter_adapted_sentences,
    segmentation_metadata,
)
from crf.utils.artifacts import atomic_writer
from crf.utils.config import PipelineConfig
from crf.utils.paths import validate_output_path
from crf.utils.resources import MemoryMonitor, check_memory
from crf.utils.serialization import (
    ValidationError,
    config_sha256,
    json_bytes,
    source_path_value,
    text_value,
)

logger = logging.getLogger(__name__)
FINGERPRINT_VERSION = 1
AUDIT_VERSION = 1
_REASONS = (
    "known_source",
    "paragraph_chars",
    "sentence_chars",
    "sentence_tokens",
    "gold_overlap",
    "known_mwt",
    "literature_duplicate",
)
_LIMITATIONS = (
    "Exact fingerprints do not detect case/punctuation revisions or near-duplicate passages.",
    "Known-source rules cover established provenance only; unresolved source separation remains a corpus-readiness gate.",
    "MWT rules cover train_gold surfaces only; no MWT expansion is performed.",
)


@dataclass(frozen=True)
class KnownSourceExclusion:
    source_path: str
    held_out_split: Literal["val_gold", "test_gold"]
    evidence: str


@dataclass(frozen=True)
class EligibleSentence:
    sentence: RawSentence
    sentence_fingerprint: str


@dataclass(frozen=True)
class ContaminationExclusion:
    source: InputIdentity
    reason: Literal["known_source", "gold_overlap", "known_mwt", "literature_duplicate"]
    sentence_id: str | None
    sentence_ordinal: int | None
    paragraph_ordinal: int | None
    source_span: tuple[int, int] | None
    characters: int | None
    tokens: int | None
    sentence_fingerprint: str | None
    matched_splits: tuple[str, ...] = ()
    matched_mwt: str | None = None
    duplicate_of: str | None = None
    evidence: str | None = None


type AuditEvent = EligibleSentence | ContaminationExclusion | RawExclusion


@dataclass(frozen=True)
class AuditReport:
    schema_version: int
    fingerprint_version: int
    preprocessing: SegmentationMetadata
    config_sha256: str
    inputs: tuple[InputIdentity, ...]
    known_source_exclusions: tuple[KnownSourceExclusion, ...]
    gold_sentence_count: int
    mwt_pattern_count: int
    accepted_sentence_count: int
    accepted_token_count: int
    exclusion_counts: tuple[tuple[str, int], ...]
    excluded_source_bytes: int
    excluded_characters: int
    corpus_sha256: str
    limitations: tuple[str, ...] = _LIMITATIONS

    def to_dict(self) -> dict[str, object]:
        """JSON-ready metadata for later guarded run publication."""
        result = asdict(self)
        result["inputs"] = [source.to_dict() for source in self.inputs]
        result["known_source_exclusions"] = [
            asdict(rule) for rule in self.known_source_exclusions
        ]
        result["exclusion_counts"] = dict(self.exclusion_counts)
        result["limitations"] = list(self.limitations)
        return result


def sentence_fingerprint(
    text: str, *, normalizer_version: int = 1, version: int = 1
) -> str:
    """Hash a case/punctuation-preserving normalized whitespace-collapsed copy."""
    if type(version) is not int or version != FINGERPRINT_VERSION:
        raise ValidationError(f"Unsupported fingerprint version: {version!r}")
    normalized = " ".join(normalize_form(text, version=normalizer_version).split())
    if not normalized:
        raise ValidationError("Fingerprint text must contain nonwhitespace content")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _rules(
    rules: Sequence[KnownSourceExclusion], sources: tuple[InputIdentity, ...]
) -> tuple[KnownSourceExclusion, ...]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise ValidationError("Known-source rules must be an ordered sequence")
    paths = {
        source.source_path for source in sources if source.source_role == "literature"
    }
    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, KnownSourceExclusion):
            raise ValidationError("Expected KnownSourceExclusion records")
        path = source_path_value(rule.source_path)
        if path not in paths or path in seen:
            raise ValidationError(f"Unknown or duplicate known-source path: {path}")
        if rule.held_out_split not in ("val_gold", "test_gold"):
            raise ValidationError("Known sources must reference val_gold or test_gold")
        if not text_value(rule.evidence, "source evidence").strip():
            raise ValidationError("Source evidence must not be whitespace-only")
        seen.add(path)
    return tuple(sorted(rules, key=lambda rule: rule.source_path))


def _empty_index(stream: BinaryIO) -> None:
    """The lifecycle helper reserves a fresh empty index, never an existing DB."""
    if stream.read(1):
        raise ValidationError("Expected a fresh empty audit index")


class ContaminationAudit:
    """Single-use state owned by contamination_audit(); not a reusable DB cache."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        config: PipelineConfig,
        root: Path,
        inputs: tuple[InputIdentity, ...],
        rules: tuple[KnownSourceExclusion, ...],
        monitor: MemoryMonitor | None = None,
    ) -> None:
        self._connection = connection
        self._config = config
        self._root = root
        self._inputs = inputs
        self._rules = rules
        self._monitor = monitor
        self._metadata = segmentation_metadata(
            adapter_version=config.preprocessing.tokenizer_adapter_version,
            normalizer_version=config.preprocessing.normalizer_version,
        )
        self._config_sha256 = config_sha256(config)
        self._counts = dict.fromkeys(_REASONS, 0)
        self._gold_count = self._accepted_count = self._accepted_tokens = 0
        self._excluded_bytes = self._excluded_chars = 0
        self._started = self._complete = self._closed = False
        self._corpus = hashlib.sha256(
            json_bytes(
                {
                    "schema_version": AUDIT_VERSION,
                    "fingerprint_version": FINGERPRINT_VERSION,
                    "preprocessing": asdict(self._metadata),
                    "config_sha256": self._config_sha256,
                    "inputs": [source.to_dict() for source in inputs],
                    "known_source_exclusions": [asdict(rule) for rule in rules],
                },
                sort_keys=True,
            )
        )

    def _build(self) -> None:
        self._checkpoint()
        db = self._connection
        db.execute("PRAGMA cache_size = -2048")
        db.execute("PRAGMA temp_store = FILE")
        db.execute(
            "CREATE TABLE gold (fingerprint TEXT, role TEXT, PRIMARY KEY (fingerprint, role)) WITHOUT ROWID"
        )
        db.execute(
            "CREATE TABLE mwt (tokens BLOB PRIMARY KEY, first TEXT, count INTEGER, surface TEXT) WITHOUT ROWID"
        )
        db.execute("CREATE INDEX mwt_head ON mwt (first, count)")
        db.execute(
            "CREATE TABLE seen (fingerprint TEXT PRIMARY KEY, sentence_id TEXT) WITHOUT ROWID"
        )
        for source in self._inputs:
            if source.source_role not in GOLD_ROLES:
                continue
            for audit in iter_gold_audit(
                source, self._config, self._root, monitor=self._monitor
            ):
                self._checkpoint()
                fingerprint = sentence_fingerprint(
                    audit.surface_text,
                    normalizer_version=self._metadata.normalizer_version,
                )
                db.execute(
                    "INSERT OR IGNORE INTO gold VALUES (?, ?)",
                    (fingerprint, source.source_role),
                )
                self._gold_count += 1
                if source.source_role == "train_gold":
                    for row in audit.surface_rows:
                        self._checkpoint()
                        if (
                            not isinstance(row.token_id, tuple)
                            or row.token_id[1] != "-"
                        ):
                            continue
                        forms = tuple(
                            form
                            for sentence in iter_adapted_sentences(
                                row.form,
                                source_offset=1,
                                adapter_version=self._metadata.tokenizer_adapter_version,
                                normalizer_version=self._metadata.normalizer_version,
                            )
                            for form in sentence.normalized_forms
                        )
                        if not forms:
                            raise ValidationError(
                                f"{source.source_path}: empty MWT surface"
                            )
                        db.execute(
                            "INSERT OR IGNORE INTO mwt VALUES (?, ?, ?, ?)",
                            (json_bytes(list(forms)), forms[0], len(forms), row.form),
                        )
            db.commit()
            self._checkpoint()

    def _checkpoint(self) -> None:
        check_memory(
            self._monitor, expected_budget_mib=self._config.resources.memory_budget_mib
        )

    def _match_mwt(self, sentence: RawSentence) -> str | None:
        forms = sentence.feature_forms
        for index, first in enumerate(forms):
            self._checkpoint()
            candidates = self._connection.execute(
                "SELECT tokens, count, surface FROM mwt WHERE first=? AND count<=? ORDER BY tokens",
                (first, len(forms) - index),
            )
            for encoded, count, surface in candidates:
                if encoded == json_bytes(list(forms[index : index + count])):
                    return str(surface)
        return None

    def _decision(
        self, sentence: RawSentence
    ) -> EligibleSentence | ContaminationExclusion:
        self._checkpoint()
        fingerprint = sentence_fingerprint(
            sentence.text.surface_text,
            normalizer_version=self._metadata.normalizer_version,
        )
        db = self._connection
        splits = tuple(
            role
            for (role,) in db.execute(
                "SELECT role FROM gold WHERE fingerprint=? ORDER BY role",
                (fingerprint,),
            )
        )
        mwt = None if splits else self._match_mwt(sentence)
        previous = (
            None
            if splits or mwt
            else db.execute(
                "SELECT sentence_id FROM seen WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        )
        if splits or mwt or previous:
            reason: Literal["gold_overlap", "known_mwt", "literature_duplicate"]
            reason = (
                "gold_overlap"
                if splits
                else "known_mwt"
                if mwt
                else "literature_duplicate"
            )
            return ContaminationExclusion(
                sentence.source,
                reason,
                sentence.sentence_id,
                sentence.sentence_ordinal,
                sentence.paragraph_ordinal,
                sentence.text.source_span,
                len(sentence.text.surface_text),
                len(sentence.feature_forms),
                fingerprint,
                splits,
                mwt,
                None if previous is None else previous[0],
            )
        db.execute(
            "INSERT INTO seen VALUES (?, ?)", (fingerprint, sentence.sentence_id)
        )
        self._accepted_count += 1
        self._accepted_tokens += len(sentence.feature_forms)
        self._corpus.update(json_bytes([sentence.sentence_id, fingerprint]))
        self._corpus.update(b"\n")
        return EligibleSentence(sentence, fingerprint)

    def iter_events(self) -> Iterator[AuditEvent]:
        """Filter before inference; an early close or failure never completes audit."""
        if self._closed or self._started:
            raise ValidationError("Contamination audit is closed or already consumed")
        self._started = True
        self._checkpoint()
        rules = {rule.source_path: rule for rule in self._rules}
        for source in self._inputs:
            self._checkpoint()
            if source.source_role != "literature":
                continue
            rule = rules.get(source.source_path)
            if rule is not None:
                exclusion = ContaminationExclusion(
                    source,
                    "known_source",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (rule.held_out_split,),
                    evidence=rule.evidence,
                )
                self._excluded_bytes += source.size_bytes
                self._record_exclusion(exclusion)
                yield exclusion
                continue
            for raw in iter_literature_events(
                source,
                config=self._config,
                repository_root=self._root,
                monitor=self._monitor,
            ):
                event = raw if isinstance(raw, RawExclusion) else self._decision(raw)
                if not isinstance(event, EligibleSentence):
                    self._record_exclusion(event)
                self._checkpoint()
                yield event
            self._connection.commit()
        if (
            tuple(
                iter_input_identities(self._config, self._root, monitor=self._monitor)
            )
            != self._inputs
        ):
            raise ValidationError("Corpus inputs changed during contamination audit")
        self._connection.commit()
        self._checkpoint()
        self._complete = True

    def _record_exclusion(self, event: RawExclusion | ContaminationExclusion) -> None:
        self._counts[event.reason] += 1
        self._excluded_chars += event.characters or 0
        logger.warning(
            "%s: contamination preprocessing excluded %s",
            event.source.source_path,
            event.reason,
        )

    def report(self) -> AuditReport:
        """Expose completion metadata only after all iterator/source checks pass."""
        if not self._complete or self._closed:
            raise ValidationError("Contamination audit is incomplete or closed")
        if self._monitor is not None and self._monitor.finished:
            self._monitor.ensure_success()
        else:
            self._checkpoint()
        count = self._connection.execute("SELECT COUNT(*) FROM mwt").fetchone()
        return AuditReport(
            AUDIT_VERSION,
            FINGERPRINT_VERSION,
            self._metadata,
            self._config_sha256,
            self._inputs,
            self._rules,
            self._gold_count,
            count[0],
            self._accepted_count,
            self._accepted_tokens,
            tuple(self._counts.items()),
            self._excluded_bytes,
            self._excluded_chars,
            self._corpus.hexdigest(),
        )


@contextmanager
def contamination_audit(
    config: PipelineConfig,
    repository_root: Path,
    *,
    index_path: Path,
    known_source_exclusions: Sequence[KnownSourceExclusion] = (),
    monitor: MemoryMonitor | None = None,
) -> Iterator[ContaminationAudit]:
    """Build a fresh guarded disk index; failures leave only incomplete intermediates.

    The caller supplies a fresh data/interim/<run_id>/*.sqlite3 location, consumes
    iter_events(), and persists report().to_dict() through normal run publication.
    No existing output is overwritten and no completion descriptor is published.
    """
    check_memory(monitor, expected_budget_mib=config.resources.memory_budget_mib)
    root = repository_root.resolve()
    destination = validate_output_path(index_path, root)
    relative = destination.relative_to(root)
    if (
        relative.parts[:2] != ("data", "interim")
        or len(relative.parts) != 4
        or relative.suffix != ".sqlite3"
    ):
        raise ValidationError("Audit index must be data/interim/<run_id>/*.sqlite3")
    inputs = tuple(iter_input_identities(config, root, monitor=monitor))
    rules = _rules(known_source_exclusions, inputs)
    # Reserve through the race-safe artifact helper before SQLite may open it.
    with atomic_writer(destination, root, validate=_empty_index):
        pass
    connection = sqlite3.connect(destination)
    audit: ContaminationAudit | None = None
    try:
        audit = ContaminationAudit(connection, config, root, inputs, rules, monitor)
        audit._build()
        yield audit
    finally:
        if audit is not None:
            audit._closed = True
        connection.close()
