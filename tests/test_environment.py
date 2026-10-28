"""Offline compatibility checks using tiny synthetic data and tmp_path."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pytest
from sklearn_crfsuite import CRF

from crf.features.surface import sentence_features


@pytest.mark.integration
def test_crf_fit_marginals_and_serialization(tmp_path: Path) -> None:
    """Requires locked sklearn-crfsuite/joblib; no network or production corpus."""
    features = [
        sentence_features(["Кіт", "спить"]),
        sentence_features(["Пес", "біжить"]),
    ]
    model = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=20)
    model.fit(features, [["NOUN", "VERB"], ["NOUN", "VERB"]])
    assert [list(labels) for labels in model.predict(features)] == [
        ["NOUN", "VERB"],
        ["NOUN", "VERB"],
    ]
    before = model.predict_marginals(features)
    for sentence in before:
        for marginal in sentence:
            assert sum(marginal.values()) == pytest.approx(1.0)
            assert all(0.0 <= value <= 1.0 for value in marginal.values())
    bundle = tmp_path / "compatibility.joblib"
    joblib.dump(model, bundle)
    restored = joblib.load(bundle)
    assert list(restored.predict(features)[0]) == ["NOUN", "VERB"]
    after = restored.predict_marginals(features)
    assert len(after) == len(before)
    for original, loaded in zip(before, after, strict=True):
        assert len(original) == len(loaded)
        for expected, actual in zip(original, loaded, strict=True):
            assert actual == expected


def test_installed_imports_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import crf; from crf.features.surface import sentence_features; print(sentence_features(['Ї'])[0]['word.lower'])",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert result.stdout.strip() == "ї"
    assert result.stderr == ""


def test_preview_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/preview_tokenization.py"),
            "--text",
            "Привіт, Київ!",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert result.stderr == ""
    assert [token["text"] for token in json.loads(result.stdout)[0]["tokens"]] == [
        "Привіт",
        ",",
        "Київ",
        "!",
    ]
