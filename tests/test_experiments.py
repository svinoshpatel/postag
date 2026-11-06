"""Specification-derived frozen ranking and strict-schema boundary cases."""

from dataclasses import replace
from typing import Any

import pytest

from crf.evaluation import ModelReference
from crf.experiments import Trial, final_rank, parse_trial
from crf.utils.serialization import ValidationError


def final(**changes: Any) -> Trial:
    return replace(
        Trial(
            "run",
            "final_000_000",
            "final",
            0.95,
            (("c1", 0.1), ("c2", 0.1)),
            "a" * 64,
            "fitted",
            ModelReference("final_000_000", "b" * 64, "c" * 64),
            0.9,
            0.8,
            None,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "left,right,tokens_left,tokens_right,winner",
    [
        (final(), final(accuracy=0.9000000000000001), 0, 100, "right"),
        (final(macro_f1=0.7), final(), 0, 100, "right"),
        (final(), final(threshold=0.98), 1, 2, "left"),
        (final(), final(threshold=0.98), 2, 2, "right"),
        (final(parameters=(("c1", 0.01), ("c2", 1.0))), final(), 2, 2, "left"),
        (final(parameters=(("c1", 0.1), ("c2", 0.01))), final(), 2, 2, "left"),
    ],
)
def test_prescribed_final_tie_breakers(
    left: Trial, right: Trial, tokens_left: int, tokens_right: int, winner: str
) -> None:
    ranked = sorted(
        [(left, tokens_left, "left"), (right, tokens_right, "right")],
        key=lambda row: final_rank(row[0], row[1]),
    )
    assert ranked[0][2] == winner


def raw_trial() -> dict[str, object]:
    return dict(
        schema_version=1,
        run_id="run",
        trial_id="final_000_000",
        stage="final",
        threshold=0.95,
        crf_parameters={"c1": 0.1, "c2": 0.1},
        training_dataset_sha256="a" * 64,
        status="fitted",
        model_ref=dict(
            candidate_id="final_000_000",
            model_sha256="b" * 64,
            metadata_sha256="c" * 64,
        ),
        validation_accuracy=0.9,
        validation_macro_f1=0.8,
        error=None,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("validation_accuracy", True),
        ("threshold", float("nan")),
        ("threshold", True),
        ("status", "unknown"),
        ("model_ref", None),
        ("validation_macro_f1", None),
        ("training_dataset_sha256", "bad"),
        ("unexpected", 1),
        ("stage", "test"),
        ("run_id", "../escape"),
    ],
)
def test_strict_trial_schema(field: str, value: object) -> None:
    raw = raw_trial()
    raw[field] = value
    with pytest.raises(ValidationError):
        parse_trial(raw)


def test_cannot_rank_incomplete_final() -> None:
    with pytest.raises(ValidationError):
        final_rank(final(threshold=None), 0)
