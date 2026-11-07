"""M9 offline CLI workflows.

Requires installed package, locked CRFsuite/joblib/NumPy/PyYAML. In-process
cases control RSS and gate confidence only; child-process smoke uses real RSS
and requires process inspection. All inputs/outputs are temporary; no network.
"""

import importlib.util
import json
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from crf import bootstrap, experiments, final_evaluation, inference, pipeline
from crf.models import bundles, reuse, sampling, search
from crf.models.confidence import ConfidenceCandidate, SentenceConfidence
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryMonitor

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def local_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline workflow attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_network)
    for module in (
        pipeline,
        experiments,
        final_evaluation,
        inference,
        bundles,
        search,
        reuse,
        sampling,
    ):
        monkeypatch.setattr(
            module,
            "MemoryMonitor",
            lambda budget: MemoryMonitor(
                budget, sampler=lambda: 1, interval_seconds=60
            ),
        )


@pytest.fixture
def workflow(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> tuple[Path, PipelineConfig]:
    for role, forms in (
        ("train", ("Кіт", "спить")),
        ("val", ("Пес", "біжить")),
        ("test", ("Птах", "летить")),
    ):
        content = ""
        for sentence in range(1, 4):
            content += f"# sent_id = {role}-{sentence}\n"
            if sentence == 1:
                content += "# newdoc id = one\n"
            elif sentence == 3:
                content += "# newdoc id = two\n"
            content += f"# text = {' '.join(forms)}\n"
            for token, (form, tag) in enumerate(
                zip(forms, ("NOUN", "VERB"), strict=True), 1
            ):
                content += f"{token}\t{form}\t_\t{tag}\t_\t_\t0\troot\t_\t_\n"
            content += "\n"
        (artifact_repository / f"data/raw/gold/{role}.conllu").write_text(
            content, encoding="utf-8"
        )
    config = replace(
        artifact_config,
        training=replace(
            artifact_config.training,
            parameters=replace(artifact_config.training.parameters, max_iterations=3),
        ),
        evaluation=replace(artifact_config.evaluation, bootstrap_resamples=25),
    )
    return artifact_repository, config


def cli(name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"e2e_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    return module


def configuration(root: Path, config: PipelineConfig) -> Path:
    path = root / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("mode", ["augmented", "empty_mapped"])
def test_complete_cli_workflow(
    workflow: tuple[Path, PipelineConfig],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    root, config = workflow
    if mode == "empty_mapped":
        config = replace(
            config,
            self_training=replace(config.self_training, pseudo_to_gold_ratio=0.0),
            tag_mapping=tuple(
                (tag, "SERVICE") for tag in ("ADP", "CCONJ", "PART", "SCONJ")
            ),
        )
    else:
        original_confidence = pipeline.iter_scored_candidates

        def accepted(*args: Any, **kwargs: Any) -> Any:
            for candidate in original_confidence(*args, **kwargs):
                yield ConfidenceCandidate(
                    candidate.sentence,
                    SentenceConfidence(
                        candidate.confidence.labels,
                        (1.0,) * len(candidate.confidence.labels),
                    ),
                )

        monkeypatch.setattr(pipeline, "iter_scored_candidates", accepted)
    path = configuration(root, config)
    originals = {
        p: p.read_bytes() for p in (root / "data/raw").rglob("*") if p.is_file()
    }
    observed_models = []
    original_fit = pipeline.fit_candidate

    def fit(records: Any, **kwargs: Any) -> Any:
        if observed_models:
            assert observed_models[-1]._tagger is None
        materialized = tuple(records)
        assert all(r.source_role in {"train_gold", "literature"} for r in materialized)
        assert any(r.source_role == "train_gold" for r in materialized)
        result = original_fit(materialized, **kwargs)
        observed_models.append(result.model)
        return result

    monkeypatch.setattr(search, "fit_candidate", fit)
    monkeypatch.setattr(pipeline, "fit_candidate", fit)
    original_loader = pipeline.iter_gold_records

    def loader(source: Any, *args: Any, **kwargs: Any) -> Any:
        assert source.source_role in {"train_gold", "val_gold"}
        yield from original_loader(source, *args, **kwargs)

    monkeypatch.setattr(pipeline, "iter_gold_records", loader)
    command = cli("run_pipeline.py", monkeypatch)
    prediction = cli("predict.py", monkeypatch)
    monkeypatch.chdir(root)
    assert command.main(["develop", "--config", str(path), "--seed", "731"]) == 0
    developed = json.loads(capsys.readouterr().out)
    run_id = developed["run_id"]
    experiment = experiments.load_development(root, run_id)
    assert experiment.config.seed == 731
    trials = [
        json.loads(line)
        for line in (root / "reports" / run_id / "trials.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert len(trials) == 36 and len({t["trial_id"] for t in trials}) == 36
    assert [t["stage"] for t in trials].count("baseline") == 9
    assert [t["stage"] for t in trials].count("final") == 27
    assert len(observed_models) == (36 if mode == "augmented" else 9)
    assert all(m._tagger is None and m.modelfile.name is None for m in observed_models)
    assert sum(t["status"] == "reused" for t in trials) == (
        0 if mode == "augmented" else 27
    )
    if mode == "empty_mapped":
        assert (
            experiment.selection.final
            == experiment.selection.gate
            == experiment.selection.gold_control
        )
    else:
        assert experiment.selection.final != experiment.selection.gold_control
    history = {
        p: p.read_bytes()
        for tree in ("reports", "models", "data/processed")
        for p in (root / tree / run_id).rglob("*")
        if p.is_file()
    }

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No training after development")

    monkeypatch.setattr(search, "fit_candidate", forbidden)
    monkeypatch.setattr(pipeline, "fit_candidate", forbidden)
    assert command.main(["evaluate", "--run-id", run_id, "--seed", "42"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    completion = root / "reports" / run_id / "evaluation/completion.json"
    outputs = json.loads(completion.read_bytes())["outputs"]
    assert set(outputs) == {
        "gate.jsonl",
        "final.jsonl",
        "gold_control.jsonl",
        "metrics.json",
        "comparison.json",
        "units.jsonl",
    }
    monkeypatch.setattr(final_evaluation, "load_bundle", forbidden)
    monkeypatch.setattr(bootstrap, "paired_bootstrap", forbidden)
    assert command.main(["evaluate", "--run-id", run_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    text = root / "raw.txt"
    text.write_text("Кіт спить\n\nКіт спить", encoding="utf-8")
    bundle = (
        root
        / "models"
        / run_id
        / "candidates"
        / experiment.selection.final.candidate_id
    )
    assert prediction.main(["--model", str(bundle), "--input", str(text)]) == 0
    predicted = json.loads(capsys.readouterr().out)
    assert predicted["sentence_count"] == 2 and predicted["token_count"] == 4
    assert predicted["label_space"] == (
        "UD_UPOS" if mode == "augmented" else "mapped_target_labels"
    )
    assert originals == {p: p.read_bytes() for p in originals}
    assert history == {p: p.read_bytes() for p in history}
    # Deliberate temporary-artifact corruption must be rejected by the same CLI.
    if mode == "augmented":
        (bundle / "model.joblib").write_bytes(b"invalid checksum")
    else:
        value = json.loads(experiment.run_path.read_bytes())
        value["schema_version"] = True
        experiment.run_path.write_text(json.dumps(value), encoding="utf-8")
    old_completion = completion.read_bytes()
    assert command.main(["evaluate", "--run-id", run_id]) == 1
    assert capsys.readouterr().out == ""
    assert completion.read_bytes() == old_completion


def test_failed_development_never_evaluates(
    workflow: tuple[Path, PipelineConfig],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, config = workflow
    original = pipeline.iter_scored_candidates

    def accepted(*args: Any, **kwargs: Any) -> Any:
        for candidate in original(*args, **kwargs):
            yield ConfidenceCandidate(
                candidate.sentence,
                SentenceConfidence(
                    candidate.confidence.labels,
                    (1.0,) * len(candidate.confidence.labels),
                ),
            )

    monkeypatch.setattr(pipeline, "iter_scored_candidates", accepted)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("controlled final bundle failure")

    monkeypatch.setattr(pipeline, "save_bundle", fail)
    command = cli("run_pipeline.py", monkeypatch)
    monkeypatch.chdir(root)
    assert command.main(["develop", "--config", str(configuration(root, config))]) == 1
    failed = json.loads(capsys.readouterr().out)
    run_id = failed["run_id"]
    trials = [
        json.loads(line)
        for line in (root / "reports" / run_id / "trials.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert len(trials) == 36
    assert [t["status"] for t in trials] == ["fitted"] * 9 + ["failed"] + [
        "not_executed"
    ] * 26
    assert command.main(["evaluate", "--run-id", run_id]) == 1
    assert capsys.readouterr().out == ""
    assert not (root / "reports" / run_id / "evaluation").exists()


def test_real_process_cli_smoke(workflow: tuple[Path, PipelineConfig]) -> None:
    root, config = workflow
    config = replace(
        config, self_training=replace(config.self_training, pseudo_to_gold_ratio=0.0)
    )
    scripts = Path(__file__).resolve().parents[1] / "scripts"

    def run(name: str, arguments: list[str]) -> dict[str, Any]:
        result = subprocess.run(
            [sys.executable, str(scripts / name), *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=120,
        )
        return json.loads(result.stdout)

    developed = run(
        "run_pipeline.py", ["develop", "--config", str(configuration(root, config))]
    )
    evaluated = run("run_pipeline.py", ["evaluate", "--run-id", developed["run_id"]])
    assert evaluated["status"] == "complete"
    selection = json.loads(
        (root / "models" / developed["run_id"] / "selected.json").read_bytes()
    )
    raw = root / "input.txt"
    raw.write_text("Птах летить", encoding="utf-8")
    result = run(
        "predict.py",
        [
            "--model",
            str(
                root
                / "models"
                / developed["run_id"]
                / "candidates"
                / selection["final"]["candidate_id"]
            ),
            "--input",
            str(raw),
        ],
    )
    assert result["status"] == "complete" and result["token_count"] == 2
