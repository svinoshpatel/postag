"""Disk-backed deterministic whole-sentence sampling and frozen training datasets."""

import hashlib
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from crf.data.contamination import AuditReport, sentence_fingerprint
from crf.data.identity import sampling_priority
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.data.segmentation import segmentation_metadata
from crf.models.confidence import ConfidenceCandidate
from crf.models.selection_diagnostics import selection_summary
from crf.utils.artifacts import _parent_fd, atomic_writer, publish_bytes
from crf.utils.config import PipelineConfig, parse_config
from crf.utils.paths import validate_output_path
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    config_sha256,
    json_bytes,
    jsonl_bytes,
    sha256_value,
)


@dataclass(frozen=True)
class FrozenDataset:
    threshold: float
    path: Path
    sha256: str
    gold_tokens: int
    token_cap: int
    pseudo_sentences: int
    pseudo_tokens: int


def _pseudo_record(
    candidate: ConfidenceCandidate, config: PipelineConfig, gate_sha256: str
) -> SentenceRecord:
    raw, score = candidate.sentence, candidate.confidence
    if raw.preprocessing != segmentation_metadata(
        adapter_version=config.preprocessing.tokenizer_adapter_version,
        normalizer_version=config.preprocessing.normalizer_version,
    ):
        raise ValidationError("Candidate preprocessing disagrees with configuration")
    text = raw.text
    fingerprint = sentence_fingerprint(
        text.surface_text, normalizer_version=config.preprocessing.normalizer_version
    )
    count = len(text.original_forms)
    record = SentenceRecord(
        1,
        "pseudo",
        raw.sentence_id,
        raw.source.source_path,
        raw.source.source_sha256,
        raw.source.source_role,
        raw.sentence_ordinal,
        raw.paragraph_ordinal,
        text.surface_text,
        "original",
        tuple(range(1, count + 1)),
        text.original_forms,
        text.normalized_forms,
        text.token_spans,
        None,
        score.labels,
        None,
    ).to_dict()
    record["pseudo"] = {
        "token_confidences": list(score.token_confidences),
        "min_confidence": score.min_confidence,
        "sentence_fingerprint": fingerprint,
        "sampling_priority": sampling_priority(config.seed, fingerprint),
        "gate_model_sha256": gate_sha256,
    }
    return parse_sentence_record(
        record,
        config=config,
        source=raw.source,
        decoded_source_length=text.source_span[1],
    )


def freeze_datasets(
    gold_records: Iterable[SentenceRecord],
    candidates: Iterable[ConfidenceCandidate],
    *,
    config: PipelineConfig,
    repository_root: Path,
    run_id: str,
    gate_model_sha256: str,
    all_scored: bool = False,
    audit_report: Callable[[], AuditReport] | None = None,
) -> tuple[FrozenDataset, ...]:
    """Consume each input once; freeze each threshold once for all subsequent fits.

    Candidates must have passed contamination/deduplication. Duplicate inputs fail.
    Original gold record validation belongs to its producer. No fitting occurs.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValidationError("Invalid sampling run ID")
    sha256_value(gate_model_sha256, "gate model checksum")
    if (
        parse_config(
            config.to_dict(), repository_root=repository_root, resolve_paths=False
        )
        != config
    ):
        raise ValidationError("Noncanonical sampling configuration")
    root = repository_root.resolve()
    directory = root / "data/interim" / run_id / "sampling"
    validate_output_path(directory, root)
    with _parent_fd(directory.relative_to(root), root, create=True) as parent:
        os.mkdir(directory.name, mode=0o700, dir_fd=parent)
    results: list[FrozenDataset] = []
    summaries: list[dict[str, object]] = []
    if type(all_scored) is not bool:
        raise ValidationError("all_scored must be boolean")
    with closing(sqlite3.connect(directory / "candidates.sqlite3")) as db:
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-2048")
            db.execute(
                "CREATE TABLE gold (ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE, body BLOB)"
            )
            db.execute(
                "CREATE TABLE pool (priority TEXT, fingerprint TEXT UNIQUE, id TEXT UNIQUE, confidence REAL, tokens INTEGER, body BLOB, source TEXT)"
            )
            db.execute("CREATE INDEX priority_order ON pool(priority, fingerprint)")
            db.execute("CREATE TABLE pos (fingerprint TEXT, label TEXT, n INTEGER)")
            db.execute("CREATE TABLE selected (fingerprint TEXT PRIMARY KEY)")
            upstream: dict[str, object] | None = None
            with MemoryMonitor(config.resources.memory_budget_mib) as preparation:
                gold_tokens = 0
                for ordinal, record in enumerate(gold_records):
                    preparation.checkpoint()
                    if (
                        record.record_type != "gold"
                        or record.source_role != "train_gold"
                        or record.source_path != config.data.train_gold.as_posix()
                        or not record.token_ids
                        or record.original_upos is None
                        or len(record.feature_forms) != len(record.token_ids)
                        or config.label_space.map_upos(record.original_upos)
                        != record.target_labels
                    ):
                        raise ValidationError(
                            "Sampling requires declared aligned training gold"
                        )
                    db.execute(
                        "INSERT INTO gold VALUES (?, ?, ?)",
                        (ordinal, record.sentence_id, jsonl_bytes(record.to_dict())),
                    )
                    gold_tokens += len(record.token_ids)
                if not gold_tokens:
                    raise ValidationError("Empty gold training dataset")
                numerator, denominator = Fraction(
                    str(config.self_training.pseudo_to_gold_ratio)
                ).as_integer_ratio()
                cap = numerator * gold_tokens // denominator
                for candidate in candidates:
                    preparation.checkpoint()
                    record = _pseudo_record(candidate, config, gate_model_sha256)
                    pseudo = record.pseudo
                    assert pseudo is not None
                    if not all_scored and pseudo.min_confidence < min(
                        config.self_training.thresholds
                    ):
                        raise ValidationError(
                            "Minimum-threshold pool contains confidence rejects; use all_scored=True"
                        )
                    db.execute(
                        "INSERT INTO pool VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            pseudo.sampling_priority,
                            pseudo.sentence_fingerprint,
                            record.sentence_id,
                            pseudo.min_confidence,
                            len(record.token_ids),
                            jsonl_bytes(record.to_dict())
                            if pseudo.min_confidence
                            >= min(config.self_training.thresholds)
                            else None,
                            record.source_path,
                        ),
                    )
                    db.executemany(
                        "INSERT INTO pos VALUES (?, ?, ?)",
                        (
                            (pseudo.sentence_fingerprint, label, n)
                            for label, n in Counter(record.target_labels).items()
                        ),
                    )
                if audit_report is not None:
                    audit = audit_report()
                    if audit.config_sha256 != config_sha256(config):
                        raise ValidationError("Audit configuration mismatch")
                    if all_scored:
                        total, tokens = db.execute(
                            "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM pool"
                        ).fetchone()
                        if (total, tokens) != (
                            audit.accepted_sentence_count,
                            audit.accepted_token_count,
                        ):
                            raise ValidationError(
                                "Audit and scored stream counts disagree"
                            )
                    upstream = audit.to_dict()
                db.commit()
            preparation.ensure_success()
            for index, threshold in enumerate(sorted(config.self_training.thresholds)):
                destination = (
                    root / "data/processed" / run_id / f"extended_{index:03d}.jsonl"
                )
                db.execute("DELETE FROM selected")
                digest = hashlib.sha256()
                selected_tokens = selected_sentences = 0
                writer_monitor = MemoryMonitor(config.resources.memory_budget_mib)

                def validate(
                    stream: BinaryIO,
                    successful: MemoryMonitor = writer_monitor,
                    expected: Callable[[], bytes] = digest.digest,
                ) -> None:
                    successful.ensure_success()
                    actual = hashlib.sha256()
                    while chunk := stream.read(config.resources.read_buffer_bytes):
                        actual.update(chunk)
                    if actual.digest() != expected():
                        raise ValidationError("Frozen dataset checksum mismatch")

                with atomic_writer(destination, root, validate=validate) as stream:
                    with writer_monitor:
                        for (body,) in db.execute(
                            "SELECT body FROM gold ORDER BY ordinal"
                        ):
                            writer_monitor.checkpoint()
                            stream.write(body)
                            digest.update(body)
                        for fingerprint, tokens, body in db.execute(
                            "SELECT fingerprint, tokens, body FROM pool WHERE confidence >= ? ORDER BY priority, fingerprint",
                            (threshold,),
                        ):
                            writer_monitor.checkpoint()
                            if selected_tokens + tokens > cap:
                                continue
                            stream.write(body)
                            digest.update(body)
                            db.execute(
                                "INSERT INTO selected VALUES (?)", (fingerprint,)
                            )
                            selected_tokens += tokens
                            selected_sentences += 1
                        summaries.append(selection_summary(db, threshold, cap))
                results.append(
                    FrozenDataset(
                        threshold,
                        destination,
                        digest.hexdigest(),
                        gold_tokens,
                        cap,
                        selected_sentences,
                        selected_tokens,
                    )
                )
        except sqlite3.IntegrityError as exc:
            raise ValidationError("Duplicate sampling sentence or fingerprint") from exc
        finally:
            db.close()
    report = dict(
        schema_version=1,
        run_id=run_id,
        config_sha256=config_sha256(config),
        confidence_denominator="all_scored" if all_scored else "minimum_threshold_pool",
        upstream_audit=upstream,
        thresholds=summaries,
    )
    with MemoryMonitor(config.resources.memory_budget_mib) as report_monitor:
        content = json_bytes(report)
    report_monitor.ensure_success()

    def validate_report(stream: BinaryIO) -> None:
        report_monitor.ensure_success()
        if stream.read() != content:
            raise ValidationError("Selection diagnostics changed")

    publish_bytes(
        root / "reports" / run_id / "pseudo_selection.json",
        content,
        root,
        validate=validate_report,
    )
    return tuple(results)
