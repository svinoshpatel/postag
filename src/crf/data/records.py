"""Immutable sentence schema and strict, context-bound validation factories."""

from dataclasses import asdict, dataclass
from typing import Literal, cast

from crf.data.identity import (
    GOLD_ROLES,
    InputIdentity,
    SourceRole,
    parse_input_identity,
    sampling_priority,
    sentence_id,
)
from crf.data.labels import TagMappingError
from crf.utils.config import ConfigError, PipelineConfig
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
class PseudoMetadata:
    token_confidences: tuple[float, ...]
    min_confidence: float
    sentence_fingerprint: str
    sampling_priority: str
    gate_model_sha256: str


@dataclass(frozen=True)
class SentenceRecord:
    """Validated factories create records; only feature_forms goes to features."""

    schema_version: int
    record_type: Literal["gold", "pseudo"]
    sentence_id: str
    source_path: str
    source_sha256: str
    source_role: SourceRole
    sentence_ordinal: int
    paragraph_ordinal: int | None
    surface_text: str
    surface_text_origin: Literal["original", "reconstructed"]
    token_ids: tuple[int, ...]
    original_forms: tuple[str, ...]
    normalized_forms: tuple[str, ...]
    token_spans: tuple[tuple[int, int] | None, ...]
    original_upos: tuple[str, ...] | None
    target_labels: tuple[str, ...]
    pseudo: PseudoMetadata | None

    @property
    def feature_forms(self) -> tuple[str, ...]:
        """Label/provenance-free input for the existing surface feature extractor."""
        return self.normalized_forms

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("token_ids", "original_forms", "normalized_forms", "target_labels"):
            result[key] = list(getattr(self, key))
        result["token_spans"] = [
            None if span is None else list(span) for span in self.token_spans
        ]
        result["original_upos"] = (
            None if self.original_upos is None else list(self.original_upos)
        )
        if self.pseudo is not None:
            result["pseudo"]["token_confidences"] = list(self.pseudo.token_confidences)
        return result


def _strings(value: object, field: str, count: int) -> tuple[str, ...]:
    result = tuple(text_value(item, field) for item in array(value, field))
    if len(result) != count:
        raise ValidationError(f"{field}: must align with {count} token IDs")
    return result


def _spans(
    value: object, count: int, decoded_source_length: int | None
) -> tuple[tuple[int, int] | None, ...]:
    if decoded_source_length is not None:
        integer(decoded_source_length, "decoded_source_length")
    result: list[tuple[int, int] | None] = []
    previous_end = 0
    for item in array(value, "token_spans"):
        if item is None:
            result.append(None)
            continue
        pair = array(item, "token span")
        if len(pair) != 2:
            raise ValidationError("token span: expected [start, end]")
        start, end = (integer(item, "token span offset") for item in pair)
        if (
            decoded_source_length is None
            or not previous_end <= start < end <= decoded_source_length
        ):
            raise ValidationError(
                "token span: missing source length, out of bounds, reversed, or overlapping"
            )
        previous_end = end
        result.append((start, end))
    if len(result) != count:
        raise ValidationError("token_spans: must align with token IDs")
    return tuple(result)


def _pseudo(value: object, count: int, seed: int) -> PseudoMetadata:
    raw = object_fields(value, set(PseudoMetadata.__dataclass_fields__), "pseudo")
    confidences = tuple(
        probability(item, "token_confidences")
        for item in array(raw["token_confidences"], "token_confidences")
    )
    if len(confidences) != count:
        raise ValidationError("token_confidences: must align with token IDs")
    minimum = probability(raw["min_confidence"], "min_confidence")
    if minimum != min(confidences):
        raise ValidationError("min_confidence: must equal exact token minimum")
    fingerprint = sha256_value(raw["sentence_fingerprint"], "sentence_fingerprint")
    priority = sha256_value(raw["sampling_priority"], "sampling_priority")
    if priority != sampling_priority(seed, fingerprint):
        raise ValidationError("sampling_priority: inconsistent with seed/fingerprint")
    return PseudoMetadata(
        confidences,
        minimum,
        fingerprint,
        priority,
        sha256_value(raw["gate_model_sha256"], "gate_model_sha256"),
    )


def parse_sentence_record(
    value: object,
    *,
    config: PipelineConfig,
    source: InputIdentity,
    decoded_source_length: int | None = None,
    final_evaluation: bool = False,
) -> SentenceRecord:
    """Validate schema and cross-field/source integrity without opening any files."""
    raw = object_fields(value, set(SentenceRecord.__dataclass_fields__), "sentence")
    if integer(raw["schema_version"], "schema_version") != 1:
        raise ValidationError("Unsupported sentence schema_version")
    parse_input_identity(source.to_dict(), config)
    if type(final_evaluation) is not bool:
        raise ValidationError("final_evaluation: expected boolean")
    if source.source_role == "test_gold" and not final_evaluation:
        raise ValidationError("test_gold records are restricted to final evaluation")
    for key in ("source_path", "source_sha256", "source_role"):
        if raw[key] != getattr(source, key) or type(raw[key]) is not str:
            raise ValidationError(f"{key}: disagrees with input identity")
    ordinal = integer(raw["sentence_ordinal"], "sentence_ordinal", 1)
    identifier = sha256_value(raw["sentence_id"], "sentence_id")
    if identifier != sentence_id(
        source.source_role, source.source_path, source.source_sha256, ordinal
    ):
        raise ValidationError("sentence_id: inconsistent provenance")
    kind = text_value(raw["record_type"], "record_type")
    if kind not in ("gold", "pseudo"):
        raise ValidationError("record_type: expected gold or pseudo")
    ids = tuple(
        integer(item, "token_ids", 1) for item in array(raw["token_ids"], "token_ids")
    )
    if not ids or ids != tuple(range(1, len(ids) + 1)):
        raise ValidationError("token_ids: expected nonempty consecutive IDs from 1")
    count = len(ids)
    original = _strings(raw["original_forms"], "original_forms", count)
    normalized = _strings(raw["normalized_forms"], "normalized_forms", count)
    labels = _strings(raw["target_labels"], "target_labels", count)
    try:
        label_space = config.label_space
    except ConfigError as exc:
        raise ValidationError(str(exc)) from exc
    spans = _spans(raw["token_spans"], count, decoded_source_length)
    surface = text_value(raw["surface_text"], "surface_text")
    origin = text_value(raw["surface_text_origin"], "surface_text_origin")
    if origin not in ("original", "reconstructed"):
        raise ValidationError("surface_text_origin: expected original or reconstructed")
    if kind == "gold":
        if (
            source.source_role not in GOLD_ROLES
            or raw["paragraph_ordinal"] is not None
            or raw["pseudo"] is not None
        ):
            raise ValidationError(
                "gold: requires gold source, null paragraph_ordinal and null pseudo"
            )
        upos = _strings(raw["original_upos"], "original_upos", count)
        try:
            mapped = label_space.map_upos(upos)
        except TagMappingError as exc:
            raise ValidationError(str(exc)) from exc
        if mapped != labels:
            raise ValidationError("target_labels: disagree with mapped original_upos")
        paragraph, pseudo = None, None
    else:
        if (
            source.source_role != "literature"
            or raw["original_upos"] is not None
            or origin != "original"
        ):
            raise ValidationError(
                "pseudo: requires literature source, null original_upos and original surface text"
            )
        paragraph = integer(raw["paragraph_ordinal"], "paragraph_ordinal", 1)
        upos = None
        pseudo = _pseudo(raw["pseudo"], count, config.seed)
    try:
        label_space.validate_targets(labels)
    except TagMappingError as exc:
        raise ValidationError(str(exc)) from exc
    return SentenceRecord(
        1,
        cast(Literal["gold", "pseudo"], kind),
        identifier,
        source.source_path,
        source.source_sha256,
        source.source_role,
        ordinal,
        paragraph,
        surface,
        cast(Literal["original", "reconstructed"], origin),
        ids,
        original,
        normalized,
        spans,
        upos,
        labels,
        pseudo,
    )
