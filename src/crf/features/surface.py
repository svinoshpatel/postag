"""Version 1 surface features; no corpus statistics or annotations are accepted."""

import unicodedata
from collections.abc import Sequence

from crf.data.normalization import (
    NORMALIZER_VERSION,
    NormalizationError,
    lexical_form,
    normalize_forms,
)
from crf.data.normalization import normalize_form as normalize_form
from crf.features.schema import (
    AFFIX_LIMITS,
    CONTEXT_OFFSETS,
    FEATURE_SCHEMA_VERSION,
    FeatureSchemaError,
    validate_feature_version,
    validate_sentence_features,
)
from crf.features.schema import FeatureDict as FeatureDict
from crf.features.schema import FeatureValue as FeatureValue


def _shape(form: str) -> FeatureDict:
    return {
        "word.lower": lexical_form(form),
        "is_title": form.istitle(),
        "is_uppercase": form.isupper(),
        "is_lowercase": form.islower(),
        "has_hyphen": any(char in form for char in "-‐‑"),
        "has_digits": any(char.isdigit() for char in form),
        "is_punct": all(unicodedata.category(char).startswith("P") for char in form),
    }


def sentence_features(
    forms: Sequence[str],
    *,
    schema_version: int = FEATURE_SCHEMA_VERSION,
    normalizer_version: int = NORMALIZER_VERSION,
) -> list[FeatureDict]:
    """Return aligned dictionaries with sentence-bounded ±2 shape context."""
    validate_feature_version(schema_version)
    try:
        normalized = normalize_forms(forms, version=normalizer_version)
    except NormalizationError as exc:
        raise FeatureSchemaError(f"forms: {exc}") from exc
    shapes = tuple(_shape(form) for form in normalized)
    result: list[FeatureDict] = []
    for index, form in enumerate(normalized):
        features: FeatureDict = {"bias": 1.0, **shapes[index]}
        lexical = lexical_form(form)
        for kind, maximum in AFFIX_LIMITS:
            for length in range(1, min(maximum, len(lexical)) + 1):
                features[f"{kind}.{length}"] = (
                    lexical[:length] if kind == "prefix" else lexical[-length:]
                )
        for offset in CONTEXT_OFFSETS:
            neighbor = index + offset
            prefix = f"{offset:+d}:"
            if neighbor < 0:
                features[prefix + "BOS"] = True
            elif neighbor >= len(normalized):
                features[prefix + "EOS"] = True
            else:
                features.update(
                    {prefix + key: value for key, value in shapes[neighbor].items()}
                )
        result.append(features)
    return validate_sentence_features(result, schema_version=schema_version)
