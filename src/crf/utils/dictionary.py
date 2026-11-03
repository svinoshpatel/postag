"""Pinned auxiliary dictionary retrieval, separate from the CRF pipeline."""

import hashlib
import logging
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, BinaryIO
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from crf.utils.paths import reject_symlink_ancestors
from crf.utils.serialization import (
    ValidationError,
    loads_json,
    sha256_value,
    text_value,
)

logger = logging.getLogger(__name__)
_DIRECTORY = Path("data/dict_uk")
_CHUNK_BYTES = 1024 * 1024


def _https_url(value: object) -> str:
    url = text_value(value, "url")
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise ValidationError("url: whitespace and control characters are forbidden")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or "\\" in url
    ):
        raise ValidationError("url: expected credential-free HTTPS URL")
    # Accessing port also rejects malformed or out-of-range values.
    _ = parts.port
    return url


@dataclass(frozen=True)
class DictionarySource:
    release: str
    filename: str
    url: str
    sha256: str

    @classmethod
    def from_json(cls, value: object) -> DictionarySource:
        """Validate required fields; preserve compatibility with extra metadata."""
        if not isinstance(value, dict):
            raise ValidationError("Dictionary source must be a JSON object")
        release = text_value(value.get("release"), "release")
        filename = text_value(value.get("filename"), "filename")
        if (
            filename in {".", ".."}
            or any(char in filename for char in "/\\\x00")
            or not filename.strip()
        ):
            raise ValidationError("filename: expected one safe basename")
        return cls(
            release,
            filename,
            _https_url(value.get("url")),
            sha256_value(value.get("sha256"), "sha256"),
        )


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@contextmanager
def _directory_fd(root: Path) -> Iterator[int]:
    """Anchor all file operations in the existing nonsymlink dictionary folder."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in _DIRECTORY.parts:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _regular_file(name: str, directory_fd: int) -> Iterator[BinaryIO]:
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
    )
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValidationError(f"Expected a regular file: {name}")
        yield stream


def _sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def fetch_dictionary(
    repository_root: Path, *, source_path: Path = _DIRECTORY / "source.json"
) -> Path:
    """Verify an existing dictionary or stream and publish a fresh pinned file.

    The manifest must be a direct child of data/dict_uk. This function never
    creates directories, replaces files or writes raw pipeline inputs.
    """
    root = repository_root.resolve()
    manifest = root / source_path
    reject_symlink_ancestors(manifest, root)
    directory = root / _DIRECTORY
    if manifest.parent.resolve() != directory or manifest.name in {".", ".."}:
        raise ValidationError("Dictionary manifest must be directly under data/dict_uk")
    with _directory_fd(root) as directory_fd:
        with _regular_file(manifest.name, directory_fd) as stream:
            source = DictionarySource.from_json(loads_json(stream.read()))
        destination = directory / source.filename
        try:
            with _regular_file(source.filename, directory_fd) as stream:
                existing_hash = _sha256(stream)
        except FileNotFoundError:
            pass
        else:
            if existing_hash != source.sha256:
                raise ValidationError(
                    f"Existing dictionary checksum is incorrect: {destination}. "
                    "Move it aside before downloading again."
                )
            logger.info("Verified existing dictionary: %s", destination)
            return destination

        temporary = f".download-{uuid.uuid4().hex}"
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(descriptor, "w+b") as stream:
                opener = build_opener(_HTTPSRedirectHandler())
                with opener.open(source.url, timeout=60) as response:
                    while chunk := response.read(_CHUNK_BYTES):
                        stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                if _sha256(stream) != source.sha256:
                    raise ValidationError(
                        "Downloaded dictionary checksum differs from the pinned release"
                    )
                os.link(
                    temporary,
                    source.filename,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            logger.info("Downloaded and verified %s: %s", source.release, destination)
            return destination
        finally:
            os.unlink(temporary, dir_fd=directory_fd)
