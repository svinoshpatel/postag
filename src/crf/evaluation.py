"""Aligned validation metrics; no test-label access or model inference."""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import BinaryIO, cast

from crf.data.normalization import lexical_form
from crf.data.records import SentenceRecord
from crf.utils.artifacts import publish_bytes
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    config_sha256,
    integer,
    json_bytes,
    loads_json,
    object_fields,
    sha256_value,
    text_value,
)


@dataclass(frozen=True)
class ModelReference:
    candidate_id: str
    model_sha256: str
    metadata_sha256: str

    def to_dict(self) -> dict[str, object]:
        if not re.fullmatch(
            r"[A-Za-z0-9_-]+", text_value(self.candidate_id, "candidate_id")
        ):
            raise ValidationError("Invalid candidate_id")
        return dict(
            candidate_id=self.candidate_id,
            model_sha256=sha256_value(self.model_sha256, "model_sha256"),
            metadata_sha256=sha256_value(self.metadata_sha256, "metadata_sha256"),
        )


@dataclass(frozen=True)
class SentencePrediction:
    sentence_id: str
    token_ids: tuple[int, ...]
    normalized_forms: tuple[str, ...]
    predicted_labels: tuple[str, ...]


@dataclass(frozen=True)
class GoldVocabulary:
    words: frozenset[str]
    effective_config_sha256: str


def gold_vocabulary(
    records: Iterable[SentenceRecord], config: PipelineConfig
) -> GoldVocabulary:
    """Build once from supervised training records, never from augmentation."""
    words: set[str] = set()
    count = 0
    for record in records:
        if (
            record.record_type != "gold"
            or record.source_role != "train_gold"
            or record.source_path != config.data.train_gold.as_posix()
        ):
            raise ValidationError("OOV vocabulary requires train_gold only")
        if not record.feature_forms:
            raise ValidationError("Empty supervised sentence")
        words.update(lexical_form(form) for form in record.feature_forms)
        count += 1
    if not count:
        raise ValidationError("Empty supervised vocabulary input")
    return GoldVocabulary(frozenset(words), config_sha256(config))


def _primary(matrix: list[list[int]], inventory: tuple[str, ...]) -> dict[str, object]:
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(inventory)))
    per_tag = {}
    for i, label in enumerate(inventory):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        per_tag[label] = dict(
            precision=tp / predicted if predicted else 0.0,
            recall=tp / support if support else 0.0,
            f1=2 * tp / (support + predicted) if support + predicted else 0.0,
            support=support,
        )
    return dict(
        token_count=total,
        correct_token_count=correct,
        accuracy=correct / total,
        macro_f1=sum(tag["f1"] for tag in per_tag.values()) / len(inventory),
        per_tag=per_tag,
        confusion_matrix=matrix,
    )


def _diagnostic(total: int, correct: int) -> dict[str, object]:
    return dict(
        token_count=total,
        correct_token_count=correct,
        accuracy=correct / total if total else None,
    )


def score_validation(
    gold: Iterable[SentenceRecord],
    predictions: Iterable[SentencePrediction],
    *,
    config: PipelineConfig,
    vocabulary: GoldVocabulary,
    model_ref: ModelReference,
) -> dict[str, object]:
    """Pool token counts with shared gold-derived masks; reject every alignment gap."""
    if vocabulary.effective_config_sha256 != config_sha256(config):
        raise ValidationError("OOV vocabulary configuration mismatch")
    inventory = config.target_inventory
    indices = {label: i for i, label in enumerate(inventory)}
    matrix = [[0] * len(inventory) for _ in inventory]
    seen: set[str] = set()
    oov_total = oov_correct = nonpunct_total = nonpunct_correct = 0
    for record, prediction in zip_longest(gold, predictions):
        if record is None or prediction is None:
            raise ValidationError("Missing or extra sentence prediction")
        if (
            record.record_type != "gold"
            or record.source_role != "val_gold"
            or record.source_path != config.data.val_gold.as_posix()
        ):
            raise ValidationError("Validation scoring requires val_gold only")
        if record.sentence_id in seen:
            raise ValidationError("Duplicate evaluation sentence")
        seen.add(record.sentence_id)
        count = len(record.token_ids)
        if (
            not count
            or record.original_upos is None
            or any(
                len(values) != count
                for values in (
                    record.feature_forms,
                    record.target_labels,
                    record.original_upos,
                )
            )
            or prediction.sentence_id != record.sentence_id
            or prediction.token_ids != record.token_ids
            or prediction.normalized_forms != record.feature_forms
            or len(prediction.predicted_labels) != count
        ):
            raise ValidationError("Misaligned sentence/token prediction")
        if config.label_space.map_upos(record.original_upos) != record.target_labels:
            raise ValidationError("Gold target mapping mismatch")
        config.label_space.validate_targets(prediction.predicted_labels)
        for form, upos, target, label in zip(
            record.feature_forms,
            record.original_upos,
            record.target_labels,
            prediction.predicted_labels,
            strict=True,
        ):
            matrix[indices[target]][indices[label]] += 1
            correct = int(target == label)
            if lexical_form(form) not in vocabulary.words:
                oov_total += 1
                oov_correct += correct
            if upos != "PUNCT":
                nonpunct_total += 1
                nonpunct_correct += correct
    if not seen:
        raise ValidationError("Empty validation split")
    return dict(
        model_ref=model_ref.to_dict(),
        sentence_count=len(seen),
        **_primary(matrix, inventory),
        oov=_diagnostic(oov_total, oov_correct),
        non_punctuation=_diagnostic(nonpunct_total, nonpunct_correct),
    )


def _close(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_close(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_close(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if type(expected) is float:
        return (
            type(actual) in (int, float)
            and 0 <= cast(float, actual) <= 1
            and math.isfinite(cast(float, actual))
            and math.isclose(
                cast(float, actual), expected, rel_tol=1e-12, abs_tol=1e-12
            )
        )
    return type(actual) is type(expected) and actual == expected


def validate_metric_report(value: object, config: PipelineConfig) -> dict[str, object]:
    """Validate validation-report schema and derive all primary values from counts."""
    raw = object_fields(
        value,
        {"schema_version", "run_id", "split", "target_inventory", "models"},
        "metrics",
    )
    if (
        integer(raw["schema_version"], "schema_version") != 1
        or raw["split"] != "val_gold"
    ):
        raise ValidationError("Expected version-1 validation report")
    if not isinstance(raw["run_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", raw["run_id"]
    ):
        raise ValidationError("Invalid metric run_id")
    inventory = config.target_inventory
    if raw["target_inventory"] != list(inventory):
        raise ValidationError("Metric target inventory mismatch")
    models = raw["models"]
    if not isinstance(models, dict) or len(models) != 1:
        raise ValidationError("Validation report requires exactly one trial")
    for trial, entry in models.items():
        if not isinstance(trial, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", trial):
            raise ValidationError("Invalid trial ID")
        model = object_fields(
            entry,
            {
                "model_ref",
                "sentence_count",
                "token_count",
                "correct_token_count",
                "accuracy",
                "macro_f1",
                "per_tag",
                "confusion_matrix",
                "oov",
                "non_punctuation",
            },
            "model metrics",
        )
        reference = object_fields(
            model["model_ref"],
            {"candidate_id", "model_sha256", "metadata_sha256"},
            "model_ref",
        )
        ModelReference(**reference).to_dict()  # type: ignore[arg-type]
        sentences = integer(model["sentence_count"], "sentence_count", 1)
        total = integer(model["token_count"], "token_count", 1)
        if sentences > total:
            raise ValidationError("More sentences than tokens")
        matrix = model["confusion_matrix"]
        n = len(inventory)
        if (
            not isinstance(matrix, list)
            or len(matrix) != n
            or any(not isinstance(row, list) or len(row) != n for row in matrix)
        ):
            raise ValidationError("Invalid confusion matrix dimensions")
        counts = [[integer(v, "confusion count") for v in row] for row in matrix]
        if sum(map(sum, counts)) != total:
            raise ValidationError("Confusion matrix total mismatch")
        expected = _primary(counts, inventory)
        if not _close({key: model[key] for key in expected}, expected):
            raise ValidationError("Metric values disagree with confusion counts")
        correct = cast(int, expected["correct_token_count"])
        for field in ("oov", "non_punctuation"):
            diagnostic = object_fields(
                model[field], {"token_count", "correct_token_count", "accuracy"}, field
            )
            size = integer(diagnostic["token_count"], field)
            hits = integer(diagnostic["correct_token_count"], field)
            if (
                not 0 <= hits <= size <= total
                or hits > correct
                or size - hits > total - correct
                or not _close(diagnostic, _diagnostic(size, hits))
            ):
                raise ValidationError("Invalid diagnostic counts/accuracy")
    return raw


def publish_validation_report(
    metrics: dict[str, object],
    *,
    config: PipelineConfig,
    repository_root: Path,
    run_id: str,
    trial_id: str,
    monitor: MemoryMonitor,
) -> Path:
    """Publish a fresh trial report after successful monitoring; never overwrite."""
    monitor.ensure_success()
    if monitor.budget_mib != config.resources.memory_budget_mib:
        raise ValidationError("Metric monitor budget mismatch")
    report = dict(
        schema_version=1,
        run_id=run_id,
        split="val_gold",
        target_inventory=list(config.target_inventory),
        models={trial_id: metrics},
    )
    content = json_bytes(report)
    validate_metric_report(loads_json(content), config)
    path = (
        repository_root / "reports" / run_id / "validation" / trial_id / "metrics.json"
    )

    def validate(stream: BinaryIO) -> None:
        monitor.ensure_success()
        validate_metric_report(loads_json(stream.read()), config)

    publish_bytes(path, content, repository_root, validate=validate)
    return path
