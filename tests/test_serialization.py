"""Exact encoding expectations and strict streaming JSON syntax failures."""

import hashlib
import json
from pathlib import Path

import pytest

from crf.utils.config import PipelineConfig, parse_config
from crf.utils.serialization import (
    ValidationError,
    canonical_config_bytes,
    config_sha256,
    iter_jsonl,
    json_bytes,
    jsonl_bytes,
    loads_json,
    read_json,
)


def test_canonical_config(
    artifact_config: PipelineConfig, artifact_repository: Path
) -> None:
    effective = artifact_config.to_dict()
    expected = json.dumps(
        effective,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert canonical_config_bytes(artifact_config) == expected
    assert not expected.endswith(b"\n")
    assert config_sha256(artifact_config) == hashlib.sha256(expected).hexdigest()
    reordered = dict(reversed(list(effective.items())))
    assert config_sha256(
        parse_config(reordered, repository_root=artifact_repository)
    ) == config_sha256(artifact_config)
    reordered["seed"] = 43
    assert config_sha256(
        parse_config(reordered, repository_root=artifact_repository)
    ) != config_sha256(artifact_config)


def test_compact_unicode_and_newline() -> None:
    assert json_bytes(["Кіт", 1]) == '["Кіт",1]'.encode()
    assert jsonl_bytes({"text": "Кіт\nспить"}) == '{"text":"Кіт\\nспить"}\n'.encode()
    assert loads_json(jsonl_bytes({"x": [None, True, 1, 0.25]})) == {
        "x": [None, True, 1, 0.25]
    }


@pytest.mark.parametrize(
    "value",
    [
        b"\xef\xbb\xbf{}",
        b"\xff",
        '{"x":1,"x":2}',
        '{"outer":{"x":1,"x":2}}',
        "NaN",
        "Infinity",
        "-Infinity",
        "1e9999",
        '"\\ud800"',
        "{} {}",
        "",
    ],
)
def test_invalid_json(value: str | bytes) -> None:
    with pytest.raises(ValidationError):
        loads_json(value)


@pytest.mark.parametrize(
    "value",
    [
        float("inf"),
        {"x": float("nan")},
        {1: "x"},
        {"x": (1, 2)},
        {"x": Path("file")},
        "\ud800",
    ],
)
def test_invalid_encoder_values(value: object) -> None:
    with pytest.raises(ValidationError):
        json_bytes(value)


def test_jsonl_streaming_and_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"n":1}\n{"n":2}\n{"bad":')
    stream = iter_jsonl(path, lambda value: value)
    assert next(stream) == {"n": 1}
    assert next(stream) == {"n": 2}
    with pytest.raises(ValidationError, match=r"records.jsonl: record/line 3"):
        next(stream)
    path.write_bytes(b'{"n":1}')
    assert list(iter_jsonl(path, lambda value: value)) == [{"n": 1}]
    path.write_bytes(b"")
    assert list(iter_jsonl(path, lambda value: value)) == []


@pytest.mark.parametrize(
    "bad",
    [b"\n", b"{}\r\n", b"[]\n", b"\xef\xbb\xbf{}\n", b'{"x":1,"x":2}\n', b"\xff\n"],
)
def test_bad_jsonl(tmp_path: Path, bad: bytes) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"{}\n" + bad)
    stream = iter_jsonl(path, lambda value: value)
    assert next(stream) == {}
    with pytest.raises(ValidationError, match=r"bad.jsonl: record/line 2"):
        next(stream)


def test_file_and_validator_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(ValidationError, match="missing.json"):
        read_json(path)
    path.write_bytes(b'{"x":false}')
    assert read_json(path) == {"x": False}

    def reject(value: object) -> object:
        raise ValidationError("schema mismatch")

    with pytest.raises(ValidationError, match="line 1: schema mismatch"):
        list(iter_jsonl(path, reject))
    with pytest.raises(ValidationError, match="must be an object"):
        jsonl_bytes([])
