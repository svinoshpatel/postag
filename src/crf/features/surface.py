"""Version 1 surface features; no corpus statistics or annotations are accepted."""

import unicodedata
from collections.abc import Sequence

from crf.data.normalization import lexical_form
from crf.data.normalization import normalize_form as normalize_form

type FeatureValue = str | bool | float
type FeatureDict = dict[str, FeatureValue]


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


def sentence_features(forms: Sequence[str]) -> list[FeatureDict]:
    """Return aligned dictionaries with sentence-bounded ±2 shape context."""
    if isinstance(forms, str) or not forms:
        raise ValueError("a nonempty sequence of surface forms is required")
    normalized = tuple(normalize_form(form) for form in forms)
    shapes = tuple(_shape(form) for form in normalized)
    result: list[FeatureDict] = []
    for index, form in enumerate(normalized):
        features: FeatureDict = {"bias": 1.0, **shapes[index]}
        lexical = lexical_form(form)
        for kind, maximum in (("prefix", 4), ("suffix", 5)):
            for length in range(1, min(maximum, len(lexical)) + 1):
                features[f"{kind}.{length}"] = (
                    lexical[:length] if kind == "prefix" else lexical[-length:]
                )
        for offset in (-2, -1, 1, 2):
            neighbor = index + offset
            prefix = f"{offset:+d}:"
            if neighbor < 0:
                features[prefix + "BOS"] = True
            elif neighbor >= len(forms):
                features[prefix + "EOS"] = True
            else:
                features.update(
                    {prefix + key: value for key, value in shapes[neighbor].items()}
                )
        result.append(features)
    return result
