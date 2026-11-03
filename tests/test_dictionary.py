"""Offline dictionary retrieval checks; fake HTTP responses and temporary folders."""

import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from http.client import HTTPMessage, IncompleteRead
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from crf.utils import dictionary
from crf.utils.dictionary import DictionarySource, fetch_dictionary

PAYLOAD = "слово\nґанок\n".encode()


def metadata(**overrides: object) -> dict[str, object]:
    return {
        "release": "fixture-release",
        "filename": "dictionary.txt",
        "url": "https://example.invalid/dictionary.txt",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        **overrides,
    }


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "data/dict_uk"
    directory.mkdir(parents=True)
    (directory / "source.json").write_text(json.dumps(metadata()), encoding="utf-8")

    def no_network(*args: object, **kwargs: object) -> Any:
        pytest.fail("Network access is forbidden")

    monkeypatch.setattr(dictionary, "build_opener", no_network)
    return tmp_path


def response(monkeypatch: pytest.MonkeyPatch, payload: bytes = PAYLOAD) -> None:
    class Response(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert size == 1024 * 1024
            return super().read(size)

    class Opener:
        def open(self, url: str, *, timeout: int) -> Response:
            assert url == metadata()["url"]
            assert timeout == 60
            return Response(payload)

    monkeypatch.setattr(dictionary, "build_opener", lambda handler: Opener())


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "../escape", "/tmp/file", "sub/file", "sub\\file", "\x00"],
)
def test_filename_rejected(filename: str) -> None:
    with pytest.raises(ValueError, match="filename"):
        DictionarySource.from_json(metadata(filename=filename))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/file",
        "file:///tmp/file",
        "https:///file",
        "https://user@example.invalid/file",
        "https://user:password@example.invalid/file",
        "https://example.invalid:99999/file",
        "https://example.invalid/\nfile",
        "https://example.invalid\\file",
    ],
)
def test_unsafe_url_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        DictionarySource.from_json(metadata(url=url))


@pytest.mark.parametrize(
    "value", [[], {}, metadata(sha256="A" * 64), metadata(release=3)]
)
def test_invalid_metadata(value: object) -> None:
    with pytest.raises(ValueError):
        DictionarySource.from_json(value)


def test_extra_metadata_compatible() -> None:
    assert (
        DictionarySource.from_json(metadata(description="auxiliary")).filename
        == "dictionary.txt"
    )


@pytest.mark.parametrize(
    "contents", ['{"release":"a","release":"b"}', "\ufeff{}", '{"extra":NaN}']
)
def test_strict_json(repository: Path, contents: str) -> None:
    (repository / "data/dict_uk/source.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        fetch_dictionary(repository)


@pytest.mark.parametrize(
    "source",
    [
        "source.json",
        "data/raw/source.json",
        "../source.json",
        "data/dict_uk/sub/source.json",
    ],
)
def test_manifest_cannot_redirect_destination(repository: Path, source: str) -> None:
    with pytest.raises(ValueError, match="directly under"):
        fetch_dictionary(repository, source_path=Path(source))


def test_existing_verified_without_network(repository: Path) -> None:
    destination = repository / "data/dict_uk/dictionary.txt"
    destination.write_bytes(PAYLOAD)
    before = destination.stat()
    assert fetch_dictionary(repository) == destination
    assert destination.stat() == before


def test_missing_manifest_has_path_diagnostic(repository: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.json"):
        fetch_dictionary(repository, source_path=Path("data/dict_uk/missing.json"))


def test_existing_mismatch_is_preserved(repository: Path) -> None:
    destination = repository / "data/dict_uk/dictionary.txt"
    destination.write_bytes(b"existing")
    with pytest.raises(ValueError, match="Existing dictionary checksum"):
        fetch_dictionary(repository)
    assert destination.read_bytes() == b"existing"


@pytest.mark.parametrize("kind", ["symlink", "dangling", "directory", "fifo"])
def test_nonregular_destination_rejected(repository: Path, kind: str) -> None:
    destination = repository / "data/dict_uk/dictionary.txt"
    if kind == "symlink":
        destination.symlink_to(destination.parent / "source.json")
    elif kind == "dangling":
        destination.symlink_to(destination.parent / "missing")
    elif kind == "directory":
        destination.mkdir()
    else:
        os.mkfifo(destination)
    with pytest.raises((OSError, ValueError)):
        fetch_dictionary(repository)


def test_manifest_symlink_and_raw_directory_alias_rejected(repository: Path) -> None:
    directory = repository / "data/dict_uk"
    manifest = directory / "source.json"
    original = manifest.read_bytes()
    manifest.unlink()
    manifest.symlink_to(repository / "outside.json")
    with pytest.raises(ValueError, match="Symlinked"):
        fetch_dictionary(repository)
    manifest.unlink()
    directory.rmdir()
    raw = repository / "data/raw"
    raw.mkdir()
    (raw / "source.json").write_bytes(original)
    directory.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        fetch_dictionary(repository)
    assert sorted(path.name for path in raw.iterdir()) == ["source.json"]


def test_download_and_alternate_manifest(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response(monkeypatch)
    directory = repository / "data/dict_uk"
    (directory / "source.json").rename(directory / "alternate.json")
    destination = fetch_dictionary(
        repository, source_path=Path("data/dict_uk/alternate.json")
    )
    assert destination.read_bytes() == PAYLOAD
    assert sorted(path.name for path in directory.iterdir()) == [
        "alternate.json",
        "dictionary.txt",
    ]


def test_bad_checksum_cleans_temporary(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response(monkeypatch, b"incorrect")
    with pytest.raises(ValueError, match="pinned release"):
        fetch_dictionary(repository)
    assert [path.name for path in (repository / "data/dict_uk").iterdir()] == [
        "source.json"
    ]


@pytest.mark.parametrize(
    "error", [OSError("broken download"), IncompleteRead(b"partial", 10)]
)
def test_download_failure_cleans_temporary(
    repository: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    class BrokenResponse(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise error

    class Opener:
        def open(self, url: str, *, timeout: int) -> BrokenResponse:
            return BrokenResponse()

    monkeypatch.setattr(dictionary, "build_opener", lambda handler: Opener())
    with pytest.raises(type(error)):
        fetch_dictionary(repository)
    assert [path.name for path in (repository / "data/dict_uk").iterdir()] == [
        "source.json"
    ]


def test_cli_incomplete_http_response(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/fetch_dictionary.py"
    main = runpy.run_path(str(script))["main"]

    def fail(*args: object, **kwargs: object) -> Path:
        raise IncompleteRead(b"partial", 10)

    monkeypatch.setitem(main.__globals__, "fetch_dictionary", fail)
    monkeypatch.setitem(main.__globals__, "configure_logging", lambda: None)
    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "IncompleteRead" in captured.err
    assert "Traceback" not in captured.err


def test_publication_race_preserves_winner(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response(monkeypatch)
    destination = repository / "data/dict_uk/dictionary.txt"
    original_link = os.link

    def race(source: str, target: str, **kwargs: Any) -> None:
        destination.write_bytes(b"winner")
        original_link(source, target, **kwargs)

    monkeypatch.setattr(dictionary.os, "link", race)
    with pytest.raises(FileExistsError):
        fetch_dictionary(repository)
    assert destination.read_bytes() == b"winner"
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "dictionary.txt",
        "source.json",
    ]


@pytest.mark.parametrize(
    "target",
    [
        "http://example.invalid/file",
        "file:///tmp/file",
        "https://user@example.invalid/file",
    ],
)
def test_redirect_rejects_unsafe_targets(target: str) -> None:
    handler = dictionary._HTTPSRedirectHandler()
    with pytest.raises(ValueError):
        handler.redirect_request(
            Request("https://example.invalid/file"),
            io.BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            target,
        )


def test_https_redirect_allowed() -> None:
    redirected = dictionary._HTTPSRedirectHandler().redirect_request(
        Request("https://example.invalid/file"),
        io.BytesIO(),
        302,
        "Found",
        HTTPMessage(),
        "https://other.invalid/file",
    )
    assert redirected is not None
    assert redirected.full_url == "https://other.invalid/file"


@pytest.mark.parametrize(
    "arguments,expected", [(["--help"], 0), (["--source", "missing.json"], 1)]
)
def test_cli_offline(arguments: list[str], expected: int) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/fetch_dictionary.py"
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=5,
    )
    assert result.returncode == expected
    if expected:
        assert result.stdout == ""
        assert "Error:" in result.stderr
    else:
        assert "--source" in result.stdout
