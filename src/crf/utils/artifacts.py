"""Unique run directories, no-clobber publication, and guarded completion IO."""

import logging
import math
import os
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

from crf.utils.paths import (
    PathValidationError,
    reject_symlink_ancestors,
    validate_output_path,
)
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    integer,
    json_bytes,
    loads_json,
    object_fields,
    text_value,
)

logger = logging.getLogger(__name__)
_ARTIFACT_TREES = ("reports", "data/interim", "data/processed", "models")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    reports: Path
    interim: Path | None
    processed: Path | None
    models: Path | None


def _relative(path: Path, root: Path) -> Path:
    candidate = root / path
    reject_symlink_ancestors(candidate, root)
    resolved = candidate.resolve()
    if not any(
        resolved.is_relative_to(root / tree) and resolved != root / tree
        for tree in _ARTIFACT_TREES
    ) or resolved.is_relative_to((root / "data/raw").resolve()):
        raise PathValidationError(f"Invalid artifact location: {path}")
    return resolved.relative_to(root)


@contextmanager
def _parent_fd(relative: Path, root: Path, *, create: bool) -> Iterator[int]:
    """Walk directories from an anchored descriptor, never following symlinks."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in relative.parent.parts:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def allocate_run(
    repository_root: Path, *, kind: Literal["develop", "predict"] = "develop"
) -> RunPaths:
    """Reserve fresh run directories; failed partial allocation stays incomplete."""
    if kind not in ("develop", "predict"):
        raise ValueError("kind must be develop or predict")
    root = repository_root.resolve()
    for _ in range(100):
        run_id = uuid.uuid4().hex
        candidates = [root / tree / run_id for tree in _ARTIFACT_TREES]
        if any(path.exists() or path.is_symlink() for path in candidates):
            continue
        paths = candidates if kind == "develop" else candidates[:1]
        for path in paths:
            validate_output_path(path, root)
            relative = _relative(path, root)
            with _parent_fd(relative, root, create=True) as fd:
                os.mkdir(relative.name, mode=0o700, dir_fd=fd)
        logger.info("Allocated %s run %s", kind, run_id)
        return RunPaths(
            run_id,
            candidates[0],
            *(candidates[1:] if kind == "develop" else [None, None, None]),
        )
    raise FileExistsError("Could not allocate a unique run ID after 100 attempts")


@contextmanager
def atomic_writer(
    path: Path, repository_root: Path, *, validate: Callable[[BinaryIO], None]
) -> Iterator[BinaryIO]:
    """Stage, fsync, validate, then atomically link without replacing any entry."""
    root = repository_root.resolve()
    validate_output_path(path, root)
    relative = _relative(path, root)
    with _parent_fd(relative, root, create=True) as directory_fd:
        temporary = f".partial-{uuid.uuid4().hex}"
        fd = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(fd, "w+b") as stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
                with os.fdopen(os.dup(stream.fileno()), "rb") as reader:
                    reader.seek(0)
                    validate(reader)
                os.link(
                    temporary,
                    relative.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            logger.info("Published artifact %s", path)
        finally:
            os.unlink(temporary, dir_fd=directory_fd)


def publish_bytes(
    path: Path,
    content: bytes,
    repository_root: Path,
    *,
    validate: Callable[[BinaryIO], None],
) -> None:
    with atomic_writer(path, repository_root, validate=validate) as stream:
        stream.write(content)


def _completion_header(value: object, relative: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationError("Completion must be an object")
    raw = object_fields(value, set(value), "completion")
    parts = relative.parts
    valid = (
        len(parts) == 3
        and parts[0] == "reports"
        and parts[2] in {"run.json", "completion.json"}
    ) or (
        len(parts) == 4
        and parts[0] == "reports"
        and parts[2:] == ("evaluation", "completion.json")
    )
    if not valid or not re.fullmatch(r"[A-Za-z0-9_-]+", parts[1]):
        raise ValidationError("Invalid completion location/run ID")
    if (
        integer(raw.get("schema_version"), "schema_version") != 1
        or raw.get("run_id") != parts[1]
    ):
        raise ValidationError("Completion schema_version/run_id mismatch")
    peak = raw.get("peak_rss_mib")
    if type(peak) not in (int, float):
        raise ValidationError("Completion peak_rss_mib must be finite and nonnegative")
    try:
        numeric_peak = cast(int | float, peak)
        valid_peak = math.isfinite(numeric_peak) and numeric_peak >= 0
    except OverflowError:
        valid_peak = False
    if not valid_peak:
        raise ValidationError("Completion peak_rss_mib must be finite and nonnegative")
    if raw.get("status") == "complete":
        if "error" not in raw or raw["error"] is not None:
            raise ValidationError("Successful completion requires null error")
    elif raw.get("status") == "failed":
        error = object_fields(raw.get("error"), {"stage", "message"}, "error")
        text_value(error["stage"], "error.stage")
        text_value(error["message"], "error.message")
    else:
        raise ValidationError("Completion status must be complete or failed")
    return raw


def publish_completion(
    path: Path,
    value: object,
    repository_root: Path,
    *,
    validate: Callable[[object], object],
    monitor: MemoryMonitor,
) -> None:
    """Publish once after stage-specific schema/link checks and finished monitoring."""
    relative = _relative(path, repository_root.resolve())
    content = json_bytes(value)

    def validate_snapshot(stream: BinaryIO) -> None:
        snapshot = loads_json(stream.read())
        raw = _completion_header(snapshot, relative)
        if (
            type(raw.get("peak_rss_mib")) not in (int, float)
            or raw["peak_rss_mib"] != monitor.peak_rss_mib
        ):
            raise ValidationError("Completion peak RSS disagrees with monitor")
        if not monitor.finished:
            raise ValidationError("Completion requires finished monitoring")
        validate(snapshot)
        if raw["status"] == "complete":
            monitor.ensure_success()

    publish_bytes(path, content, repository_root, validate=validate_snapshot)


def load_completed[T](
    path: Path, repository_root: Path, *, validate: Callable[[object], T]
) -> T:
    """Require a valid successful descriptor; stage validator must verify its links."""
    root = repository_root.resolve()
    relative = _relative(path, root)
    try:
        with _parent_fd(relative, root, create=False) as directory_fd:
            fd = os.open(
                relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            with os.fdopen(fd, "rb") as stream:
                snapshot = loads_json(stream.read())
        raw = _completion_header(snapshot, relative)
        if raw["status"] != "complete":
            raise ValidationError("Failed run cannot authorize success")
        return validate(snapshot)
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"{path}: incomplete or invalid completion: {exc}"
        ) from exc
