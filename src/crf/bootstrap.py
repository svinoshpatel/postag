"""Prescribed paired PCG64 bootstrap over whole gold-derived units."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from crf.data.conllu import GoldSentence
from crf.utils.resources import MemoryMonitor, check_memory
from crf.utils.serialization import ValidationError, integer


@dataclass(frozen=True)
class Unit:
    unit_id: str
    sentence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return dict(
            schema_version=1, unit_id=self.unit_id, sentence_ids=list(self.sentence_ids)
        )


@dataclass(frozen=True)
class Outcome:
    sentence_id: str
    tokens: int
    gate_correct: int
    final_correct: int


def resampling_units(
    gold: tuple[GoldSentence, ...],
) -> tuple[Literal["document", "sentence"], tuple[Unit, ...], str | None]:
    groups: dict[str, list[str]] = {}
    previous: str | None = None
    complete = bool(gold)
    for sentence in gold:
        document = sentence.audit.document_id
        if document is None or (document in groups and document != previous):
            complete = False
            break
        groups.setdefault(document, []).append(sentence.audit.sentence_id)
        previous = document
    if complete:
        return (
            "document",
            tuple(Unit(members[0], tuple(members)) for members in groups.values()),
            None,
        )
    return (
        "sentence",
        tuple(Unit(s.audit.sentence_id, (s.audit.sentence_id,)) for s in gold),
        "Document grouping is missing or ambiguous; sentence fallback used",
    )


def paired_bootstrap(
    units: tuple[Unit, ...],
    outcomes: tuple[Outcome, ...],
    *,
    unit_type: Literal["document", "sentence"],
    resamples: int,
    seed: int,
    monitor: MemoryMonitor | None = None,
) -> dict[str, object]:
    integer(resamples, "resamples", 1)
    integer(seed, "bootstrap seed")
    if unit_type not in ("document", "sentence"):
        raise ValidationError("Unknown resampling unit type")
    by_id = {outcome.sentence_id: outcome for outcome in outcomes}
    if len(by_id) != len(outcomes):
        raise ValidationError("Duplicate outcome sentence")
    for outcome in outcomes:
        integer(outcome.tokens, "tokens", 1)
        for correct in (outcome.gate_correct, outcome.final_correct):
            integer(correct, "correct count")
            if correct > outcome.tokens:
                raise ValidationError("Correct count exceeds tokens")
    positions = {outcome.sentence_id: i for i, outcome in enumerate(outcomes)}
    members: list[str] = []
    stats = []
    firsts = []
    for unit in units:
        check_memory(monitor)
        if (
            not unit.sentence_ids
            or unit.unit_id != unit.sentence_ids[0]
            or any(s not in by_id for s in unit.sentence_ids)
        ):
            raise ValidationError("Invalid resampling membership")
        indices = [positions[s] for s in unit.sentence_ids]
        if indices != sorted(indices) or (
            unit_type == "sentence" and len(indices) != 1
        ):
            raise ValidationError("Invalid resampling member order/type")
        firsts.append(indices[0])
        members.extend(unit.sentence_ids)
        stats.append(
            (
                sum(by_id[s].tokens for s in unit.sentence_ids),
                sum(by_id[s].gate_correct for s in unit.sentence_ids),
                sum(by_id[s].final_correct for s in unit.sentence_ids),
            )
        )
    if (
        firsts != sorted(firsts)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
    ):
        raise ValidationError("Units must partition all sentences disjointly in order")
    report: dict[str, object] = dict(
        unit_type=unit_type,
        unit_count=len(units),
        resamples=resamples,
        seed=seed,
        method="percentile",
        percentile_interpolation="linear",
        lower_pp=None,
        upper_pp=None,
        unavailable_reason=None,
    )
    if len(units) < 2:
        report["unavailable_reason"] = "Fewer than two resampling units"
        return report
    random = np.random.Generator(np.random.PCG64(seed))
    differences = []
    for _ in range(resamples):
        check_memory(monitor)
        sampled = random.integers(0, len(units), size=len(units), dtype=np.int64)
        tokens = sum(stats[int(i)][0] for i in sampled)
        gate = sum(stats[int(i)][1] for i in sampled)
        final = sum(stats[int(i)][2] for i in sampled)
        differences.append(100 * (final / tokens - gate / tokens))
    differences.sort()

    def percentile(p: float) -> float:
        h = (resamples - 1) * p
        left, right = math.floor(h), math.ceil(h)
        return differences[left] + (differences[right] - differences[left]) * (h - left)

    report.update(lower_pp=percentile(0.025), upper_pp=percentile(0.975))
    return report
