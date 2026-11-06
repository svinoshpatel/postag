"""Tiny native frozen evaluation and cache-integrity tests; locked dependencies."""

import json
from typing import Any

import pytest

from crf import final_evaluation
from crf.evaluation import score_test
from crf.utils.serialization import ValidationError


def test_frozen_three_roles_and_cache(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation
    before = {
        p: p.read_bytes()
        for tree in ("models", "reports")
        for p in (root / tree).rglob("*")
        if p.is_file()
    }
    result = final_evaluation.evaluate_core(root, experiment.run_id)
    assert set(result.metrics["models"]) == {"gate", "final", "gold_control"}
    assert all(m["token_count"] == 6 for m in result.metrics["models"].values())
    for role in final_evaluation.ROLES:
        values = [
            json.loads(line)
            for line in (result.directory / f"{role}.jsonl").read_bytes().splitlines()
        ]
        assert len(values) == 3
        assert all(
            "target_labels" not in v and "original_upos" not in v for v in values
        )
        assert values[0]["source_role"] == "test_gold"

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cached evaluation must not infer/load native models")

    monkeypatch.setattr(final_evaluation, "load_bundle", forbidden)
    assert (
        final_evaluation.evaluate_core(root, experiment.run_id).metrics
        == result.metrics
    )
    assert before == {p: p.read_bytes() for p in before}
    assert not (result.directory / "completion.json").exists()


@pytest.mark.parametrize(
    "fault", ["prediction", "metrics", "core", "partial", "source", "alignment"]
)
def test_cache_rejects_corruption(frozen_evaluation: Any, fault: str) -> None:
    import hashlib

    root, experiment = frozen_evaluation
    result = final_evaluation.evaluate_core(root, experiment.run_id)
    if fault == "source":
        (root / experiment.config.data.test_gold).write_bytes(b"changed")
    elif fault == "partial":
        (result.directory / "core.json").unlink()
    elif fault == "core":
        (result.directory / "core.json").write_bytes(b"{}")
    elif fault in ("prediction", "alignment"):
        path = result.directory / "gate.jsonl"
        values = path.read_bytes().splitlines()
        if fault == "alignment":
            values.reverse()
            path.write_bytes(b"\n".join(values) + b"\n")
            core = json.loads((result.directory / "core.json").read_bytes())
            core["outputs"]["gate.jsonl"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            (result.directory / "core.json").write_text(
                json.dumps(core), encoding="utf-8"
            )
        else:
            path.write_bytes(b"{}\n")
    else:
        (result.directory / "metrics.json").write_bytes(b"{}")
    with pytest.raises((ValueError, OSError)):
        final_evaluation.evaluate_core(root, experiment.run_id)


def test_seed_assertion_precedes_output(frozen_evaluation: Any) -> None:
    root, experiment = frozen_evaluation
    with pytest.raises(ValidationError, match="seed"):
        final_evaluation.evaluate_core(root, experiment.run_id, seed=43)
    assert not (root / "reports" / experiment.run_id / "evaluation").exists()


def test_no_implicit_test_scoring() -> None:
    with pytest.raises(ValidationError, match="explicit"):
        score_test([], [], config=None, vocabulary=None, model_ref=None)


def test_inference_failure_never_caches_success(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("controlled inference failure")

    monkeypatch.setattr(final_evaluation, "score_sentence", fail)
    with pytest.raises(RuntimeError, match="controlled"):
        final_evaluation.evaluate_core(root, experiment.run_id)
    directory = root / "reports" / experiment.run_id / "evaluation"
    assert not (directory / "core.json").exists()
    with pytest.raises((ValueError, OSError)):
        final_evaluation.evaluate_core(root, experiment.run_id)
