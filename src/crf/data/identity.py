"""Automatic opaque-byte source identities and provenance-derived sentence IDs."""

import hashlib
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from crf.utils.config import PipelineConfig, validate_input_paths
from crf.utils.paths import reject_symlink_ancestors
from crf.utils.serialization import (
    ValidationError,
    integer,
    json_bytes,
    object_fields,
    sha256_value,
    source_path_value,
    text_value,
)

type SourceRole = Literal["train_gold", "val_gold", "test_gold", "literature"]
GOLD_ROLES: tuple[SourceRole, ...] = ("train_gold", "val_gold", "test_gold")
SOURCE_ROLES = (*GOLD_ROLES, "literature")


@dataclass(frozen=True)
class InputIdentity:
    """Exact source bytes and provenance; construct through parse/input helpers."""

    schema_version: int
    source_role: SourceRole
    source_path: str
    source_sha256: str
    size_bytes: int
    encoding: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FileDigest:
    sha256: str
    size_bytes: int


def parse_input_identity(value: object, config: PipelineConfig) -> InputIdentity:
    raw = object_fields(value, set(InputIdentity.__dataclass_fields__), "input")
    if integer(raw["schema_version"], "schema_version") != 1:
        raise ValidationError("Unsupported input schema_version")
    role = text_value(raw["source_role"], "source_role")
    if role not in SOURCE_ROLES:
        raise ValidationError("Invalid input source_role")
    path = source_path_value(raw["source_path"])
    encoding = text_value(raw["encoding"], "encoding")
    if role in GOLD_ROLES:
        if path != getattr(config.data, role).as_posix() or encoding != "utf-8":
            raise ValidationError(
                "Gold input path/encoding disagrees with configuration"
            )
    elif (
        PurePosixPath(path).parent.as_posix() != config.data.literature_dir.as_posix()
        or PurePosixPath(path).suffix != ".txt"
        or encoding != config.data.source_encoding
    ):
        raise ValidationError(
            "Literature input path/encoding disagrees with configuration"
        )
    return InputIdentity(
        1,
        role,
        path,
        sha256_value(raw["source_sha256"], "source_sha256"),
        integer(raw["size_bytes"], "size_bytes"),
        encoding,
    )


def digest_file(path: Path, *, buffer_bytes: int = 65536) -> FileDigest:
    """Hash bounded original-byte chunks and reject observed file replacement/change."""
    integer(buffer_bytes, "buffer_bytes", 1)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(buffer_bytes):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
        signatures = {
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            for stat in (before, after, current)
        }
        if len(signatures) != 1 or size != before.st_size:
            raise ValidationError(f"{path}: source changed while hashing")
    except OSError as exc:
        raise ValidationError(f"{path}: {exc}") from exc
    return FileDigest(digest.hexdigest(), size)


def sentence_id(
    source_role: SourceRole, source_path: str, source_sha256: str, sentence_ordinal: int
) -> str:
    """Hash the exact ordered JSON identity array, independently of text or labels."""
    if source_role not in SOURCE_ROLES:
        raise ValidationError("Invalid sentence source_role")
    source_path_value(source_path)
    sha256_value(source_sha256, "source_sha256")
    integer(sentence_ordinal, "sentence_ordinal", 1)
    return hashlib.sha256(
        json_bytes([source_role, source_path, source_sha256, sentence_ordinal])
    ).hexdigest()


def sampling_priority(seed: int, sentence_fingerprint: str) -> str:
    integer(seed, "seed")
    sha256_value(sentence_fingerprint, "sentence_fingerprint")
    return hashlib.sha256(
        f"{seed}:{sentence_fingerprint}".encode("utf-8", errors="strict")
    ).hexdigest()


def _source_path(
    identity: InputIdentity, config: PipelineConfig, repository_root: Path
) -> Path:
    """Validate current resolved source boundaries before any opaque byte reads."""
    parse_input_identity(identity.to_dict(), config)
    root = repository_root.resolve()
    path = root / identity.source_path
    if identity.source_role == "literature":
        reject_symlink_ancestors(path, root)
    resolved = path.resolve()
    parent = (
        root
        / (
            "data/raw/gold"
            if identity.source_role in GOLD_ROLES
            else config.data.literature_dir
        )
    ).resolve()
    if (
        not parent.is_relative_to(root)
        or not resolved.is_relative_to(parent)
        or not resolved.is_file()
    ):
        raise ValidationError(f"Invalid source location: {path}")
    if identity.source_role == "literature" and any(
        resolved.samefile(root / getattr(config.data, role)) for role in GOLD_ROLES
    ):
        raise ValidationError(f"Literature aliases gold input: {path}")
    return resolved


def verify_input_identity(
    identity: InputIdentity, config: PipelineConfig, repository_root: Path
) -> None:
    path = _source_path(identity, config, repository_root)
    actual = digest_file(path, buffer_bytes=config.resources.read_buffer_bytes)
    if (actual.sha256, actual.size_bytes) != (
        identity.source_sha256,
        identity.size_bytes,
    ):
        raise ValidationError(f"{path}: source checksum/size mismatch")


def iter_input_identities(
    config: PipelineConfig, repository_root: Path
) -> Iterator[InputIdentity]:
    """Discover and hash sources without decoding or interpreting any annotation."""
    validate_input_paths(config, repository_root)
    root = repository_root.resolve()
    sources: list[tuple[SourceRole, Path, str]] = [
        (role, getattr(config.data, role), "utf-8") for role in GOLD_ROLES
    ]
    literature = root / config.data.literature_dir
    # This holds paths only, never corpus text or parsed records.
    for path in sorted(literature.iterdir(), key=lambda path: path.name):
        if path.suffix != ".txt":
            continue
        if path.is_symlink():
            raise ValidationError(f"Symlinked literature candidate: {path}")
        if path.is_file():
            sources.append(
                ("literature", path.relative_to(root), config.data.source_encoding)
            )
    if len(sources) == 3:
        raise ValidationError(f"No eligible literature files: {literature}")
    for role, relative, encoding in sources:
        stub = InputIdentity(1, role, relative.as_posix(), "0" * 64, 0, encoding)
        path = _source_path(stub, config, root)
        digest = digest_file(path, buffer_bytes=config.resources.read_buffer_bytes)
        yield InputIdentity(
            1, role, relative.as_posix(), digest.sha256, digest.size_bytes, encoding
        )
