"""Original-byte identities, bounded hashing, and source-isolation fixtures."""

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from crf.data.identity import (
    InputIdentity,
    digest_file,
    iter_input_identities,
    parse_input_identity,
    sampling_priority,
    sentence_id,
    verify_input_identity,
)
from crf.utils.config import PipelineConfig
from crf.utils.serialization import ValidationError, jsonl_bytes, loads_json


def test_identities_order_bytes_and_verification(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
) -> None:
    assert [source.source_role for source in artifact_sources] == [
        "train_gold",
        "val_gold",
        "test_gold",
        "literature",
        "literature",
    ]
    assert [source.source_path for source in artifact_sources[-2:]] == [
        "data/raw/literature/а.txt",
        "data/raw/literature/б.txt",
    ]
    for source in artifact_sources:
        original = (artifact_repository / source.source_path).read_bytes()
        assert source.source_sha256 == hashlib.sha256(original).hexdigest()
        assert source.size_bytes == len(original)
        assert (
            parse_input_identity(
                loads_json(jsonl_bytes(source.to_dict())), artifact_config
            )
            == source
        )
        verify_input_identity(source, artifact_config, artifact_repository)
    assert (
        (artifact_repository / artifact_sources[2].source_path)
        .read_bytes()
        .startswith(b"\xff")
    )
    with pytest.raises(FrozenInstanceError):
        artifact_sources[0].size_bytes = 9


def test_exact_sentence_identity_and_priority() -> None:
    values = ["literature", "data/raw/literature/ї.txt", "a" * 64, 17]
    expected = hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert sentence_id("literature", values[1], values[2], 17) == expected
    assert sentence_id("literature", values[1], values[2], 18) != expected
    assert (
        sampling_priority(0, "b" * 64)
        == hashlib.sha256(("0:" + "b" * 64).encode()).hexdigest()
    )
    for seed in (True, -1):
        with pytest.raises(ValidationError):
            sampling_priority(seed, "b" * 64)
    with pytest.raises(ValidationError):
        sentence_id("literature", "../bad.txt", "a" * 64, 1)
    with pytest.raises(ValidationError):
        sentence_id("literature", values[1], values[2], True)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("source_role", "raw"),
        ("source_sha256", "A" * 64),
        ("source_sha256", "0" * 63),
        ("source_sha256", "z" * 64),
        ("size_bytes", True),
        ("size_bytes", -1),
        ("size_bytes", 1.0),
        ("encoding", ""),
        ("encoding", "cp1251"),
        ("source_path", "/data/raw/gold/train.conllu"),
        ("source_path", "./data/raw/gold/train.conllu"),
        ("source_path", "data//raw/gold/train.conllu"),
        ("source_path", "data/raw/../gold/train.conllu"),
        ("source_path", "data\\raw\\gold\\train.conllu"),
        ("source_path", "data/raw/gold/val.conllu"),
    ],
)
def test_bad_input_fields(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    field: str,
    bad: object,
) -> None:
    value = artifact_sources[0].to_dict() | {field: bad}
    with pytest.raises(ValidationError):
        parse_input_identity(value, artifact_config)


def test_input_exact_fields_and_config_encoding(
    artifact_config: PipelineConfig, artifact_sources: tuple[InputIdentity, ...]
) -> None:
    value = artifact_sources[0].to_dict()
    for key in value:
        with pytest.raises(ValidationError):
            parse_input_identity(
                {name: item for name, item in value.items() if name != key},
                artifact_config,
            )
    with pytest.raises(ValidationError):
        parse_input_identity(value | {"extra": None}, artifact_config)
    config = replace(
        artifact_config, data=replace(artifact_config.data, source_encoding="cp1251")
    )
    lit = artifact_sources[-1].to_dict() | {"encoding": "cp1251"}
    assert parse_input_identity(lit, config).encoding == "cp1251"
    for path in (
        "data/raw/literature/nested/book.txt",
        "data/raw/gold/book.txt",
        "data/raw/literature/book.TXT",
    ):
        with pytest.raises(ValidationError):
            parse_input_identity(lit | {"source_path": path}, config)


def test_bounded_digest_empty_and_corruption(
    tmp_path: Path,
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"\x00\xff" * 100)
    assert (
        digest_file(path, buffer_bytes=1).sha256
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    path.write_bytes(b"")
    assert digest_file(path).size_bytes == 0
    assert digest_file(path).sha256 == hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValidationError):
        digest_file(path, buffer_bytes=True)
    with pytest.raises(ValidationError):
        digest_file(tmp_path / "missing")
    with pytest.raises(ValidationError, match="checksum/size mismatch"):
        verify_input_identity(
            replace(artifact_sources[0], source_sha256="0" * 64),
            artifact_config,
            artifact_repository,
        )


def test_observed_change_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "changed"
    path.write_bytes(b"original")
    original_stat = os.fstat
    calls = 0

    def changed_stat(fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"different length")
        return original_stat(fd)

    monkeypatch.setattr(os, "fstat", changed_stat)
    with pytest.raises(ValidationError, match="source changed while hashing"):
        digest_file(path, buffer_bytes=2)


@pytest.mark.parametrize(
    "kind", ["symlink", "dangling", "gold_hardlink", "empty", "ignored"]
)
def test_literature_discovery(
    artifact_repository: Path, artifact_config: PipelineConfig, kind: str
) -> None:
    literature = artifact_repository / "data/raw/literature"
    if kind in {"symlink", "dangling"}:
        (literature / "link.txt").symlink_to(
            literature / ("а.txt" if kind == "symlink" else "missing")
        )
    elif kind == "gold_hardlink":
        os.link(
            artifact_repository / artifact_config.data.test_gold,
            literature / "alias.txt",
        )
    elif kind == "empty":
        for path in literature.iterdir():
            path.unlink()
    else:
        (literature / "nested.txt").mkdir()
        (literature / "nested.txt/hidden.txt").write_bytes(b"hidden")
        (literature / "upper.TXT").write_bytes(b"ignored")
        (literature / "readme.md").write_bytes(b"ignored")
        assert (
            len(tuple(iter_input_identities(artifact_config, artifact_repository))) == 5
        )
        return
    with pytest.raises(ValidationError):
        tuple(iter_input_identities(artifact_config, artifact_repository))
