"""No-clobber lifecycle checks use pytest temporary artifact trees only."""

import hashlib
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest

import crf.utils.artifacts as artifacts
from crf.utils.artifacts import (
    allocate_run,
    atomic_writer,
    load_completed,
    publish_bytes,
    publish_completion,
)
from crf.utils.paths import PathValidationError
from crf.utils.resources import (
    MemoryBudgetExceeded,
    MemoryMonitor,
    ResourceMonitoringError,
)
from crf.utils.serialization import (
    ValidationError,
    loads_json,
    object_fields,
)


def accept(stream: BinaryIO) -> None:
    assert stream.read() == b"artifact"


def descriptor(run_id: str, monitor: MemoryMonitor) -> dict[str, object]:
    # Foundation fixture, not a development/evaluation descriptor or real run.
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "error": None,
        "peak_rss_mib": monitor.peak_rss_mib,
        "output_sha256": hashlib.sha256(b"artifact").hexdigest(),
    }


def linked_validator(path: Path) -> Callable[[object], dict[str, object]]:
    def validate(value: object) -> dict[str, object]:
        raw = object_fields(
            value,
            {
                "schema_version",
                "run_id",
                "status",
                "error",
                "peak_rss_mib",
                "output_sha256",
            },
            "fixture",
        )
        if raw["output_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValidationError("linked checksum mismatch")
        return raw

    return validate


def finished_monitor() -> MemoryMonitor:
    monitor = MemoryMonitor(1, interval_seconds=60, sampler=lambda: 0)
    with monitor:
        pass
    return monitor


def test_allocate_development_and_prediction(tmp_path: Path) -> None:
    first = allocate_run(tmp_path)
    assert first.reports == tmp_path / "reports" / first.run_id
    assert first.interim == tmp_path / "data/interim" / first.run_id
    assert first.processed == tmp_path / "data/processed" / first.run_id
    assert first.models == tmp_path / "models" / first.run_id
    for path in (first.reports, first.interim, first.processed, first.models):
        assert path.is_dir()
        assert list(path.iterdir()) == []
    second = allocate_run(tmp_path, kind="predict")
    assert first.run_id != second.run_id
    assert second.reports.is_dir()
    assert second.interim is second.processed is second.models is None
    assert not (tmp_path / "models" / second.run_id).exists()
    with pytest.raises(ValueError):
        allocate_run(tmp_path, kind="other")


def test_collision_and_partial_run_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models/existing").mkdir(parents=True)
    (tmp_path / "models/existing/original").write_bytes(b"keep")
    ids = iter(["existing", "fresh"])
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    result = allocate_run(tmp_path)
    assert result.run_id == "fresh"
    assert (tmp_path / "models/existing/original").read_bytes() == b"keep"
    assert not (tmp_path / "reports/existing").exists()


def test_allocation_failure_leaves_no_completion(tmp_path: Path) -> None:
    (tmp_path / "data").write_bytes(b"blocking file")
    with pytest.raises(OSError):
        allocate_run(tmp_path)
    reports = list((tmp_path / "reports").iterdir())
    assert len(reports) == 1
    assert list(reports[0].iterdir()) == []


def test_atomic_publication_and_write_failure(tmp_path: Path) -> None:
    path = tmp_path / "reports/run/output.bin"
    with atomic_writer(path, tmp_path, validate=accept) as stream:
        stream.write(b"artifact")
        assert not path.exists()
    assert path.read_bytes() == b"artifact"
    assert list(path.parent.iterdir()) == [path]
    with pytest.raises(PathValidationError):
        publish_bytes(path, b"overwrite", tmp_path, validate=accept)
    failed = path.parent / "failed.bin"
    with pytest.raises(ValueError, match="body failed"):
        with atomic_writer(failed, tmp_path, validate=accept) as stream:
            stream.write(b"partial")
            raise ValueError("body failed")
    assert not failed.exists()
    assert list(path.parent.iterdir()) == [path]


def test_validation_failure_and_readonly_reader(tmp_path: Path) -> None:
    path = tmp_path / "reports/run/output.bin"

    def reject(stream: BinaryIO) -> None:
        assert stream.read() == b"artifact"
        assert not stream.writable()
        raise ValidationError("bad output")

    with pytest.raises(ValidationError, match="bad output"):
        publish_bytes(path, b"artifact", tmp_path, validate=reject)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "hardlink"])
def test_preserve_existing_entries(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "reports/run/existing"
    path.parent.mkdir(parents=True)
    original = tmp_path / "original"
    original.write_bytes(b"keep")
    if kind == "file":
        path.write_bytes(b"keep")
    elif kind == "directory":
        path.mkdir()
    elif kind == "dangling":
        path.symlink_to(tmp_path / "missing")
    else:
        os.link(original, path)
    with pytest.raises(PathValidationError):
        publish_bytes(path, b"artifact", tmp_path, validate=accept)
    assert original.read_bytes() == b"keep"
    assert path.exists() or path.is_symlink()


def test_raw_and_symlink_protection(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    original = raw / "original"
    original.write_bytes(b"keep")
    reports = tmp_path / "reports"
    reports.symlink_to(raw, target_is_directory=True)
    for path in (raw / "new", original, reports / "new"):
        with pytest.raises(PathValidationError):
            publish_bytes(path, b"artifact", tmp_path, validate=accept)
    with pytest.raises(PathValidationError):
        allocate_run(tmp_path)
    assert list(raw.iterdir()) == [original]
    assert original.read_bytes() == b"keep"


def test_concurrent_publish_only_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "reports/run/race.bin"
    barrier = threading.Barrier(2)

    def validate(stream: BinaryIO) -> None:
        barrier.wait(timeout=2)

    def publish(content: bytes) -> bool:
        try:
            publish_bytes(path, content, tmp_path, validate=validate)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, content) for content in (b"one", b"two")]
        assert sorted(future.result(timeout=3) for future in futures) == [False, True]
    assert path.read_bytes() in {b"one", b"two"}
    assert list(path.parent.iterdir()) == [path]


def test_target_appears_during_validation(tmp_path: Path) -> None:
    path = tmp_path / "reports/run/race.bin"

    def race(stream: BinaryIO) -> None:
        path.write_bytes(b"keep winner")

    with pytest.raises(FileExistsError):
        publish_bytes(path, b"artifact", tmp_path, validate=race)
    assert path.read_bytes() == b"keep winner"
    assert list(path.parent.iterdir()) == [path]


def test_completion_link_validation_and_no_clobber(tmp_path: Path) -> None:
    run = allocate_run(tmp_path, kind="predict")
    output = run.reports / "output.bin"
    publish_bytes(output, b"artifact", tmp_path, validate=accept)
    monitor = finished_monitor()
    value = descriptor(run.run_id, monitor)
    validator = linked_validator(output)
    completion = run.reports / "completion.json"
    publish_completion(completion, value, tmp_path, validate=validator, monitor=monitor)
    assert loads_json(completion.read_bytes()) == value
    assert load_completed(completion, tmp_path, validate=validator) == value
    with pytest.raises(PathValidationError):
        publish_completion(
            completion, value, tmp_path, validate=validator, monitor=monitor
        )

    # Wrong expected checksum tests integrity without modifying a published artifact.
    def reject(value: object) -> object:
        raise ValidationError("changed linked checksum")

    with pytest.raises(ValidationError, match="changed linked checksum"):
        load_completed(completion, tmp_path, validate=reject)


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "wrong"},
        {"schema_version": True},
        {"status": "partial"},
        {"error": {"stage": "x", "message": "x"}},
        {"peak_rss_mib": True},
        {"peak_rss_mib": 1},
        {"output_sha256": "wrong"},
        {"extra": 1},
    ],
)
def test_invalid_completion_never_published(
    tmp_path: Path, change: dict[str, object]
) -> None:
    run = allocate_run(tmp_path, kind="predict")
    output = run.reports / "output.bin"
    publish_bytes(output, b"artifact", tmp_path, validate=accept)
    monitor = finished_monitor()
    path = run.reports / "completion.json"
    with pytest.raises(ValidationError):
        publish_completion(
            path,
            descriptor(run.run_id, monitor) | change,
            tmp_path,
            validate=linked_validator(output),
            monitor=monitor,
        )
    assert not path.exists()
    assert list(run.reports.iterdir()) == [output]


def test_missing_partial_failed_and_budget_violation(tmp_path: Path) -> None:
    run = allocate_run(tmp_path, kind="predict")
    path = run.reports / "completion.json"
    output = run.reports / "output.bin"
    publish_bytes(output, b"artifact", tmp_path, validate=accept)
    validator = linked_validator(output)
    with pytest.raises(ValidationError, match="incomplete"):
        load_completed(path, tmp_path, validate=validator)
    (run.reports / ".partial-crash").write_bytes(b'{"status":')
    with pytest.raises(ValidationError):
        load_completed(path, tmp_path, validate=validator)
    monitor = MemoryMonitor(1, sampler=lambda: 2 * 1024 * 1024)
    with pytest.raises(MemoryBudgetExceeded):
        with monitor:
            pass
    value = descriptor(run.run_id, monitor)
    with pytest.raises(MemoryBudgetExceeded):
        publish_completion(path, value, tmp_path, validate=validator, monitor=monitor)
    assert not path.exists()
    failed = value | {
        "status": "failed",
        "error": {"stage": "memory", "message": "budget exceeded"},
    }
    publish_completion(path, failed, tmp_path, validate=validator, monitor=monitor)
    with pytest.raises(ValidationError, match="Failed run"):
        load_completed(path, tmp_path, validate=validator)
    assert loads_json(path.read_bytes()) == failed


def test_active_and_aborted_monitor_cannot_publish_success(tmp_path: Path) -> None:
    run = allocate_run(tmp_path, kind="predict")
    path = run.reports / "completion.json"
    output = run.reports / "output.bin"
    publish_bytes(output, b"artifact", tmp_path, validate=accept)
    monitor = MemoryMonitor(1, interval_seconds=60, sampler=lambda: 0)
    with pytest.raises(ValueError, match="abort"):
        with monitor:
            with pytest.raises(ValidationError, match="finished"):
                publish_completion(
                    path,
                    descriptor(run.run_id, monitor),
                    tmp_path,
                    validate=linked_validator(output),
                    monitor=monitor,
                )
            raise ValueError("abort")
    with pytest.raises(ResourceMonitoringError, match="unsuccessfully"):
        publish_completion(
            path,
            descriptor(run.run_id, monitor),
            tmp_path,
            validate=linked_validator(output),
            monitor=monitor,
        )
    assert not path.exists()


def test_fsync_failure_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reports/run/output.bin"

    def fail(fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(artifacts.os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failure"):
        publish_bytes(path, b"artifact", tmp_path, validate=accept)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


def test_corrupt_completion_and_invalid_locations(tmp_path: Path) -> None:
    monitor = finished_monitor()
    run = allocate_run(tmp_path, kind="predict")
    output = run.reports / "output.bin"
    publish_bytes(output, b"artifact", tmp_path, validate=accept)
    path = run.reports / "completion.json"
    path.write_bytes(b'{"status":')
    with pytest.raises(ValidationError, match="incomplete or invalid"):
        load_completed(path, tmp_path, validate=linked_validator(output))
    for location in (
        tmp_path / "models/run/completion.json",
        run.reports / "unknown.json",
    ):
        with pytest.raises(ValidationError, match="Invalid completion location"):
            publish_completion(
                location,
                descriptor(run.run_id, monitor),
                tmp_path,
                validate=linked_validator(output),
                monitor=monitor,
            )
