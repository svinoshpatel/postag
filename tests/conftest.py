"""Shared offline artifact fixtures; originals exist only in pytest tmp_path."""

from pathlib import Path

import pytest

from crf.data.identity import InputIdentity, iter_input_identities
from crf.utils.config import PipelineConfig, parse_config


@pytest.fixture
def artifact_repository(tmp_path: Path) -> Path:
    gold = tmp_path / "data/raw/gold"
    gold.mkdir(parents=True)
    for name in ("train", "val", "test"):
        (gold / f"{name}.conllu").write_bytes(b"\xff opaque " + name.encode())
    literature = tmp_path / "data/raw/literature"
    literature.mkdir()
    (literature / "б.txt").write_bytes(b"\xef\xbb\xbfB\r\n")
    (literature / "а.txt").write_text("Кіт спить.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def artifact_config(artifact_repository: Path) -> PipelineConfig:
    return parse_config(
        {
            "config_version": 1,
            "data": {
                "train_gold": "data/raw/gold/train.conllu",
                "val_gold": "data/raw/gold/val.conllu",
                "test_gold": "data/raw/gold/test.conllu",
                "ud_release": "offline-fixture",
            },
        },
        repository_root=artifact_repository,
    )


@pytest.fixture
def artifact_sources(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> tuple[InputIdentity, ...]:
    return tuple(iter_input_identities(artifact_config, artifact_repository))


@pytest.fixture
def frozen_evaluation(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Tiny real frozen run shared by M8 tests; no production corpus/network."""
    from dataclasses import replace

    from crf import experiments, final_evaluation, inference, pipeline
    from crf.models import bundles, reuse, sampling, search
    from crf.utils.resources import MemoryMonitor

    for module in (
        experiments,
        final_evaluation,
        inference,
        pipeline,
        bundles,
        reuse,
        sampling,
        search,
    ):
        monkeypatch.setattr(
            module,
            "MemoryMonitor",
            lambda budget: MemoryMonitor(
                budget, sampler=lambda: 1, interval_seconds=60
            ),
        )
    for name, words in (
        ("train", ("Кіт", "спить")),
        ("val", ("Пес", "біжить")),
        ("test", ("Птах", "летить")),
    ):
        content = ""
        for ordinal in range(1, 4):
            content += f"# sent_id = {name}-{ordinal}\n"
            if ordinal == 1:
                content += "# newdoc id = document-one\n"
            if ordinal == 3:
                content += "# newdoc id = document-two\n"
            content += f"# text = {' '.join(words)}\n"
            for i, (form, label) in enumerate(
                zip(words, ("NOUN", "VERB"), strict=True), 1
            ):
                content += f"{i}\t{form}\t_\t{label}\t_\t_\t0\troot\t_\t_\n"
            content += "\n"
        (artifact_repository / f"data/raw/gold/{name}.conllu").write_text(
            content, encoding="utf-8"
        )
    config = replace(
        artifact_config,
        training=replace(
            artifact_config.training,
            c1_grid=(0.1,),
            c2_grid=(0.1,),
            parameters=replace(artifact_config.training.parameters, max_iterations=3),
        ),
        self_training=replace(artifact_config.self_training, pseudo_to_gold_ratio=0.0),
        evaluation=replace(artifact_config.evaluation, bootstrap_resamples=40),
    )
    return artifact_repository, pipeline.develop(config, artifact_repository)
