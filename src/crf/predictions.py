"""Strict label-free prediction records shared by final and raw inference."""

from dataclasses import asdict, dataclass
from typing import Literal

from crf.data.identity import sentence_id
from crf.data.normalization import normalize_forms
from crf.evaluation import ModelReference
from crf.experiments import _reference, identifier
from crf.models.confidence import SentenceConfidence
from crf.utils.config import PipelineConfig
from crf.utils.serialization import (
    ValidationError,
    array,
    integer,
    object_fields,
    probability,
    sha256_value,
    text_value,
)


@dataclass(frozen=True)
class Prediction:
    schema_version: int
    run_id: str
    model_ref: ModelReference
    sentence_id: str
    source_path: str
    source_sha256: str
    source_role: Literal["val_gold", "test_gold", "raw"]
    sentence_ordinal: int
    token_ids: tuple[int, ...]
    original_forms: tuple[str, ...]
    normalized_forms: tuple[str, ...]
    token_spans: tuple[tuple[int, int] | None, ...]
    predicted_labels: tuple[str, ...]
    token_confidences: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["model_ref"] = self.model_ref.to_dict()
        for field in (
            "token_ids",
            "original_forms",
            "normalized_forms",
            "predicted_labels",
            "token_confidences",
        ):
            raw[field] = list(getattr(self, field))
        raw["token_spans"] = [
            None if span is None else list(span) for span in self.token_spans
        ]
        return raw


def parse_prediction(value: object, config: PipelineConfig) -> Prediction:
    raw = object_fields(value, set(Prediction.__dataclass_fields__), "prediction")
    if integer(raw["schema_version"], "schema_version") != 1 or raw[
        "source_role"
    ] not in ("val_gold", "test_gold", "raw"):
        raise ValidationError("Invalid prediction schema/role")
    path = text_value(raw["source_path"], "source_path")
    sha = sha256_value(raw["source_sha256"], "source_sha256")
    ordinal = integer(raw["sentence_ordinal"], "sentence_ordinal", 1)
    role: Literal["val_gold", "test_gold", "raw"] = raw["source_role"]
    sid = sha256_value(raw["sentence_id"], "sentence_id")
    if sid != sentence_id(role, path, sha, ordinal):
        raise ValidationError("Prediction identity mismatch")
    ids = tuple(integer(v, "token ID", 1) for v in array(raw["token_ids"], "token_ids"))
    if not ids or ids != tuple(range(1, len(ids) + 1)):
        raise ValidationError("Invalid prediction token IDs")

    def strings(field: str) -> tuple[str, ...]:
        values = tuple(text_value(v, field) for v in array(raw[field], field))
        if len(values) != len(ids):
            raise ValidationError("Prediction arrays must align")
        return values

    original, normalized, labels = (
        strings(f) for f in ("original_forms", "normalized_forms", "predicted_labels")
    )
    if (
        normalize_forms(original, version=config.preprocessing.normalizer_version)
        != normalized
    ):
        raise ValidationError("Prediction normalization mismatch")
    config.label_space.validate_targets(labels)
    confidences = tuple(
        probability(v, "token confidence")
        for v in array(raw["token_confidences"], "token_confidences")
    )
    SentenceConfidence(labels, confidences)
    spans: list[tuple[int, int] | None] = []
    previous = 0
    for span in array(raw["token_spans"], "token_spans"):
        if span is None:
            spans.append(None)
        else:
            pair = array(span, "token span")
            if len(pair) != 2:
                raise ValidationError("Invalid token span")
            start, end = (integer(v, "token offset") for v in pair)
            if not previous <= start < end:
                raise ValidationError("Invalid token span ordering")
            previous = end
            spans.append((start, end))
    if len(spans) != len(ids):
        raise ValidationError("Prediction spans must align")
    return Prediction(
        1,
        identifier(raw["run_id"]),
        _reference(raw["model_ref"]),
        sid,
        path,
        sha,
        role,
        ordinal,
        ids,
        original,
        normalized,
        tuple(spans),
        labels,
        confidences,
    )
