"""Independent standalone inference with tiny trusted local native bundles."""

import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crf import inference, pipeline
from crf.models.confidence import SentenceConfidence
from crf.utils.resources import MemoryBudgetExceeded, MemoryMonitor

pytestmark = pytest.mark.integration


def model_path(root: Path, experiment: Any) -> Path:
    return (
        root
        / "models"
        / experiment.run_id
        / "candidates"
        / experiment.selection.gate.candidate_id
    )


def test_repeated_low_confidence_overlap_and_fresh_runs(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation
    path = root / "input.txt"
    path.write_text("Кіт спить\n\nКіт спить\n\nКіт спить", encoding="utf-8")
    original = inference.score_sentence

    def low(*args: Any, **kwargs: Any) -> SentenceConfidence:
        result = original(*args, **kwargs)
        return SentenceConfidence(result.labels, (0.01,) * len(result.labels))

    monkeypatch.setattr(inference, "score_sentence", low)
    first = inference.predict(model_path(root, experiment), path, root)
    second = inference.predict(model_path(root, experiment), path, root)
    assert first.run_id != second.run_id
    assert first.sentence_count == second.sentence_count == 3
    assert first.token_count == 6
    rows = [
        json.loads(line) for line in first.predictions_path.read_bytes().splitlines()
    ]
    assert [r["sentence_ordinal"] for r in rows] == [1, 2, 3]
    assert all(
        r["source_role"] == "raw"
        and r["source_path"] == "input.txt"
        and r["token_confidences"] == [0.01, 0.01]
        for r in rows
    )
    assert len({r["sentence_id"] for r in rows}) == 3
    assert path.read_text(encoding="utf-8") == "Кіт спить\n\nКіт спить\n\nКіт спить"
    assert json.loads(first.completion_path.read_bytes())["status"] == "complete"


def test_standalone_copy_without_original_corpora(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation
    standalone = root / "standalone"
    standalone.mkdir()
    bundle = standalone / "bundle"
    shutil.copytree(model_path(root, experiment), bundle)
    path = root / "external.txt"
    path.write_text("Птах летить.", encoding="utf-8")
    original = Path.open

    def guarded(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.is_relative_to(root / "data/raw"):
            raise AssertionError(
                "standalone prediction must not access original corpora"
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = inference.predict(bundle, path, standalone)
    row = json.loads(result.predictions_path.read_bytes().splitlines()[0])
    assert row["source_path"] == path.resolve().as_posix()
    assert not (standalone / "data").exists()
    assert result.label_space == "UD_UPOS"


@pytest.mark.parametrize("content", ["", "\n \n"])
def test_empty_eligible_input(frozen_evaluation: Any, content: str) -> None:
    root, experiment = frozen_evaluation
    path = root / "empty.txt"
    path.write_text(content, encoding="utf-8")
    result = inference.predict(model_path(root, experiment), path, root)
    assert result.sentence_count == result.token_count == 0
    assert result.predictions_path.read_bytes() == b""
    assert json.loads(result.completion_path.read_bytes())["predictions_sha256"]


@pytest.mark.parametrize(
    "reason", ["paragraph_chars", "sentence_chars", "sentence_tokens"]
)
def test_resource_exclusions_preserve_suffix(
    frozen_evaluation: Any, reason: str
) -> None:
    root, experiment = frozen_evaluation
    limits = experiment.config.resources
    prefix = (
        "А" * (limits.max_paragraph_chars + 1)
        if reason == "paragraph_chars"
        else "А" * (limits.max_sentence_chars + 1)
        if reason == "sentence_chars"
        else " ".join(["Кіт"] * (limits.max_sentence_tokens + 1))
    )
    path = root / "large.txt"
    path.write_text(prefix + "\n\nПтах летить", encoding="utf-8")
    result = inference.predict(model_path(root, experiment), path, root)
    assert result.sentence_count == 1
    row = json.loads(result.predictions_path.read_bytes().splitlines()[0])
    assert row["original_forms"] == ["Птах", "летить"]
    assert row["token_spans"][0][0] >= len(prefix) + 2
    diagnostics = json.loads(
        (result.completion_path.parent / "diagnostics.json").read_bytes()
    )
    assert diagnostics["resource_exclusions"] == {reason: 1}
    assert diagnostics["resource_excluded_characters"] == {reason: len(prefix)}
    assert diagnostics["resource_excluded_tokens"] == {
        reason: limits.max_sentence_tokens + 1 if reason == "sentence_tokens" else None
    }
    assert diagnostics["limitations"]


def test_invalid_decoding_has_no_success(frozen_evaluation: Any) -> None:
    root, experiment = frozen_evaluation
    path = root / "invalid.txt"
    path.write_bytes(b"\xff")
    before = set((root / "reports").iterdir())
    with pytest.raises(ValueError, match="decoding"):
        inference.predict(model_path(root, experiment), path, root)
    directory = next(iter(set((root / "reports").iterdir()) - before))
    assert (
        json.loads((directory / "completion.json").read_bytes())["status"] == "failed"
    )
    assert not (directory / "predictions.jsonl").exists()


def test_source_change_refuses_success(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation
    path = root / "changing.txt"
    path.write_text("Кіт спить", encoding="utf-8")
    original = inference.score_sentence

    def change(*args: Any, **kwargs: Any) -> Any:
        path.write_text("Змінений текст", encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "score_sentence", change)
    with pytest.raises(ValueError, match="changed"):
        inference.predict(model_path(root, experiment), path, root)


def test_mapped_inventory_is_explicit(frozen_evaluation: Any) -> None:
    root, experiment = frozen_evaluation
    mapping = tuple((tag, "CUSTOM") for tag in ("ADP", "CCONJ", "PART", "SCONJ"))
    mapped = pipeline.develop(replace(experiment.config, tag_mapping=mapping), root)
    path = root / "mapped.txt"
    path.write_text("Кіт спить", encoding="utf-8")
    result = inference.predict(model_path(root, mapped), path, root)
    diagnostics = json.loads(
        (result.completion_path.parent / "diagnostics.json").read_bytes()
    )
    assert result.label_space == diagnostics["label_space"] == "mapped_target_labels"
    assert "CUSTOM" in diagnostics["target_inventory"]
    assert "ADP" not in diagnostics["target_inventory"]


def test_resource_exit_cannot_authorize_prediction(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation
    path = root / "input.txt"
    path.write_text("Кіт спить", encoding="utf-8")

    class Failed(MemoryMonitor):
        def __exit__(self, *args: Any) -> Any:
            super().__exit__(*args)
            raise MemoryBudgetExceeded("controlled raw prediction exit")

    monkeypatch.setattr(
        inference,
        "MemoryMonitor",
        lambda budget: Failed(budget, sampler=lambda: 1, interval_seconds=60),
    )
    before = set((root / "reports").iterdir())
    with pytest.raises(MemoryBudgetExceeded, match="controlled"):
        inference.predict(model_path(root, experiment), path, root)
    directory = next(iter(set((root / "reports").iterdir()) - before))
    assert (
        json.loads((directory / "completion.json").read_bytes())["status"] == "failed"
    )


def test_predict_cli(
    frozen_evaluation: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, experiment = frozen_evaluation
    script = Path(__file__).resolve().parents[1] / "scripts/predict.py"
    spec = importlib.util.spec_from_file_location("predict_cli", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.chdir(root)
    path = root / "text.txt"
    path.write_text("Кіт спить", encoding="utf-8")
    assert (
        module.main(
            ["--model", str(model_path(root, experiment)), "--input", str(path)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["sentence_count"] == 1
    assert (
        module.main(
            ["--model", str(model_path(root, experiment)), "--input", "missing.txt"]
        )
        == 1
    )
    assert capsys.readouterr().out == ""


def test_recorded_non_utf8_encoding(frozen_evaluation: Any) -> None:
    root, experiment = frozen_evaluation
    for path in (root / "data/raw/literature").glob("*.txt"):
        path.write_bytes("Сонце світить.".encode("cp1251"))
    config = replace(
        experiment.config,
        data=replace(experiment.config.data, source_encoding="cp1251"),
    )
    encoded_model = pipeline.develop(config, root)
    path = root / "encoded.txt"
    path.write_bytes("Кіт спить".encode("cp1251"))
    result = inference.predict(model_path(root, encoded_model), path, root)
    row = json.loads(result.predictions_path.read_bytes().splitlines()[0])
    assert row["original_forms"] == ["Кіт", "спить"]
    assert (
        json.loads(result.completion_path.read_bytes())["input"]["encoding"] == "cp1251"
    )


def test_boolean_input_byte_count_is_rejected(
    frozen_evaluation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, experiment = frozen_evaluation
    path = root / "one_byte.txt"
    path.write_bytes(b"B")
    original = inference.publish_completion

    def invalid(path: Path, value: Any, *args: Any, **kwargs: Any) -> Any:
        value["input"]["size_bytes"] = True
        return original(path, value, *args, **kwargs)

    monkeypatch.setattr(inference, "publish_completion", invalid)
    with pytest.raises(ValueError, match="integer"):
        inference.predict(model_path(root, experiment), path, root)


def test_excluded_volume_aggregates_without_token_guesses(
    frozen_evaluation: Any,
) -> None:
    root, experiment = frozen_evaluation
    size = experiment.config.resources.max_paragraph_chars + 3
    path = root / "two_large_paragraphs.txt"
    path.write_text(
        "А" * size + "\n\n" + "Б" * size + "\n\nКіт спить", encoding="utf-8"
    )
    result = inference.predict(model_path(root, experiment), path, root)
    report = json.loads(
        (result.completion_path.parent / "diagnostics.json").read_bytes()
    )
    assert report["resource_exclusions"] == {"paragraph_chars": 2}
    assert report["resource_excluded_characters"] == {"paragraph_chars": 2 * size}
    assert report["resource_excluded_tokens"] == {"paragraph_chars": None}
    assert result.sentence_count == 1 and result.token_count == 2
