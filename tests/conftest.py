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
