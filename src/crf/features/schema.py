"""Strict version-1 dictionary schema for sentence-local CRFsuite features."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

from crf.data.normalization import lexical_form, normalize_form

FEATURE_SCHEMA_VERSION = 1
CONTEXT_OFFSETS = (-2, -1, 1, 2)
AFFIX_LIMITS = (("prefix", 4), ("suffix", 5))
SHAPE_FLAGS = (
    "is_title",
    "is_uppercase",
    "is_lowercase",
    "has_hyphen",
    "has_digits",
    "is_punct",
)
SHAPE_KEYS = ("word.lower", *SHAPE_FLAGS)
type FeatureValue = str | bool | float
type FeatureDict = dict[str, FeatureValue]


class FeatureSchemaError(ValueError):
    """A version, input or feature dictionary violates the feature contract."""


def validate_feature_version(version: object) -> int:
    if type(version) is not int or version != FEATURE_SCHEMA_VERSION:
        raise FeatureSchemaError(f"Unsupported feature schema_version: {version!r}")
    return version


def _text(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise FeatureSchemaError(f"{context}: expected a nonempty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise FeatureSchemaError(f"{context}: invalid Unicode") from exc
    return value


def _value(key: str, value: object, context: str) -> FeatureValue:
    base = key.split(":")[-1]
    if base == "bias":
        if type(value) is not float or not math.isfinite(value) or value != 1.0:
            raise FeatureSchemaError(f"{context}: bias must be the finite float 1.0")
        return value
    if base in (*SHAPE_FLAGS, "BOS", "EOS"):
        if type(value) is not bool:
            raise FeatureSchemaError(f"{context}: expected boolean")
        if base in ("BOS", "EOS") and not value:
            raise FeatureSchemaError(f"{context}: boundary flag must be true")
        return value
    return _text(value, context)


def validate_sentence_features(
    value: object, *, schema_version: int = FEATURE_SCHEMA_VERSION
) -> list[FeatureDict]:
    """Validate exact keys/types/affixes/context and return independent dictionaries."""
    validate_feature_version(schema_version)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise FeatureSchemaError("features: expected a nonempty sentence sequence")
    rows: list[FeatureDict] = []
    for index, item in enumerate(value):
        context = f"token {index + 1}"
        if not isinstance(item, Mapping) or any(type(key) is not str for key in item):
            raise FeatureSchemaError(f"{context}: expected a mapping with string keys")
        raw = dict(cast(Mapping[str, object], item))
        lexical = _text(raw.get("word.lower"), f"{context} word.lower")
        if lexical_form(normalize_form(lexical)) != lexical:
            raise FeatureSchemaError(
                f"{context} word.lower: expected normalized lexical form"
            )
        affixes = {
            f"{kind}.{length}": lexical[:length]
            if kind == "prefix"
            else lexical[-length:]
            for kind, maximum in AFFIX_LIMITS
            for length in range(1, min(maximum, len(lexical)) + 1)
        }
        expected = {"bias", *SHAPE_KEYS, *affixes}
        for offset in CONTEXT_OFFSETS:
            neighbor = index + offset
            prefix = f"{offset:+d}:"
            if neighbor < 0:
                expected.add(prefix + "BOS")
            elif neighbor >= len(value):
                expected.add(prefix + "EOS")
            else:
                expected.update(prefix + key for key in SHAPE_KEYS)
        if raw.keys() != expected:
            raise FeatureSchemaError(
                f"{context}: missing keys {sorted(expected - raw.keys())}; "
                f"unsupported keys {sorted(raw.keys() - expected)}"
            )
        row = {key: _value(key, item, f"{context} {key}") for key, item in raw.items()}
        for key, expected_affix in affixes.items():
            if row[key] != expected_affix:
                raise FeatureSchemaError(f"{context} {key}: inconsistent affix")
        rows.append(row)
    for index, row in enumerate(rows):
        for offset in CONTEXT_OFFSETS:
            neighbor = index + offset
            if 0 <= neighbor < len(rows):
                for key in SHAPE_KEYS:
                    context_key = f"{offset:+d}:{key}"
                    if row[context_key] != rows[neighbor][key]:
                        raise FeatureSchemaError(
                            f"token {index + 1} {context_key}: disagrees with neighbor"
                        )
    return rows
