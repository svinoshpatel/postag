"""Hand-derived paired-resampling contracts and fixed seed references."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from crf import bootstrap
from crf.bootstrap import Outcome, Unit, paired_bootstrap, resampling_units
from crf.utils.serialization import ValidationError


def test_manual_pooled_repeated_units(monkeypatch: pytest.MonkeyPatch) -> None:
    draws = iter(([0, 1], [0, 0], [1, 1]))

    class Random:
        def integers(self, low: int, high: int, *, size: int, dtype: Any) -> Any:
            assert (low, high, size, dtype) == (0, 2, 2, np.int64)
            return np.array(next(draws), dtype=dtype)

    monkeypatch.setattr(bootstrap.np.random, "Generator", lambda bitgen: Random())
    result = paired_bootstrap(
        (Unit("a", ("a",)), Unit("b", ("b",))),
        (Outcome("a", 1, 1, 0), Outcome("b", 9, 0, 9)),
        unit_type="sentence",
        resamples=3,
        seed=42,
    )
    assert result["lower_pp"] == pytest.approx(-91)
    assert result["upper_pp"] == pytest.approx(99)
    assert result["unavailable_reason"] is None


def test_exact_seeded_reference() -> None:
    units = (Unit("a", ("a",)), Unit("b", ("b",)), Unit("c", ("c",)))
    outcomes = (Outcome("a", 1, 1, 0), Outcome("b", 9, 0, 9), Outcome("c", 2, 2, 0))
    random = np.random.Generator(np.random.PCG64(73))
    values = []
    for _ in range(23):
        sampled = [
            outcomes[int(i)] for i in random.integers(0, 3, size=3, dtype=np.int64)
        ]
        total = sum(s.tokens for s in sampled)
        values.append(
            100
            * (
                sum(s.final_correct for s in sampled) / total
                - sum(s.gate_correct for s in sampled) / total
            )
        )
    result = paired_bootstrap(
        units, outcomes, unit_type="sentence", resamples=23, seed=73
    )
    assert [result["lower_pp"], result["upper_pp"]] == pytest.approx(
        np.percentile(values, [2.5, 97.5], method="linear")
    )


def test_unavailable_without_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no random generator for fewer than two units")

    monkeypatch.setattr(bootstrap.np.random, "Generator", forbidden)
    result = paired_bootstrap(
        (Unit("a", ("a",)),),
        (Outcome("a", 2, 2, 0),),
        unit_type="document",
        resamples=2000,
        seed=42,
    )
    assert result["lower_pp"] is result["upper_pp"] is None
    assert result["unavailable_reason"]


@pytest.mark.parametrize(
    "documents,expected",
    [
        (["a", "a", "b"], "document"),
        ([None, "a", "a"], "sentence"),
        (["a", "b", "a"], "sentence"),
    ],
)
def test_grouping(documents: list[str | None], expected: str) -> None:
    gold = tuple(
        SimpleNamespace(audit=SimpleNamespace(document_id=d, sentence_id=str(i)))
        for i, d in enumerate(documents)
    )
    kind, units, reason = resampling_units(gold)
    assert kind == expected
    assert bool(reason) == (expected == "sentence")
    assert tuple(s for unit in units for s in unit.sentence_ids) == ("0", "1", "2")


@pytest.mark.parametrize(
    "units,outcomes",
    [
        ((Unit("a", ("a",)), Unit("a", ("a",))), (Outcome("a", 1, 1, 0),)),
        ((Unit("a", ("a",)),), (Outcome("a", 1, 1, 0), Outcome("b", 1, 1, 1))),
        ((Unit("b", ("b",)),), (Outcome("a", 1, 1, 0),)),
        ((Unit("a", ("a",)),), (Outcome("a", 1, 2, 0),)),
    ],
)
def test_invalid_partition_and_counts(units: Any, outcomes: Any) -> None:
    with pytest.raises(ValidationError):
        paired_bootstrap(units, outcomes, unit_type="sentence", resamples=2, seed=42)
