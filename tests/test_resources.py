"""Controlled process/RSS observations; no large allocations or live low-budget runs."""

import logging
import threading
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from crf.utils.logging import configure_logging
from crf.utils.resources import (
    MemoryBudgetExceeded,
    MemoryMonitor,
    ResourceMonitoringError,
    aggregate_rss_bytes,
)

MIB = 1024 * 1024


@pytest.mark.parametrize("budget", [True, 0, -1, 1.0])
def test_invalid_budget(budget: int) -> None:
    with pytest.raises(ValueError):
        MemoryMonitor(budget)


@pytest.mark.parametrize("interval", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        MemoryMonitor(1, interval_seconds=interval)


def test_threshold_peak_and_single_use() -> None:
    samples = iter([MIB, MIB // 2, 0])
    monitor = MemoryMonitor(1, interval_seconds=60, sampler=lambda: next(samples))
    with monitor:
        assert monitor.peak_rss_mib == 1.0
        monitor.checkpoint()
        with pytest.raises(ResourceMonitoringError, match="finished"):
            monitor.ensure_success()
    assert monitor.finished
    assert monitor.peak_rss_mib == 1.0
    monitor.ensure_success()
    with pytest.raises(ResourceMonitoringError, match="reused"):
        with monitor:
            pass
    with pytest.raises(ResourceMonitoringError, match="not active"):
        monitor.checkpoint()


def test_entry_failure_latched() -> None:
    monitor = MemoryMonitor(1, sampler=lambda: MIB + 1)
    with pytest.raises(MemoryBudgetExceeded):
        with monitor:
            pytest.fail("Body must not run")
    assert monitor.finished
    assert monitor.peak_rss_mib == (MIB + 1) / MIB
    with pytest.raises(MemoryBudgetExceeded):
        monitor.ensure_success()


def test_checkpoint_failure_stays_fatal_after_recovery() -> None:
    samples = iter([0, 2 * MIB, 0])
    monitor = MemoryMonitor(1, interval_seconds=60, sampler=lambda: next(samples))
    with pytest.raises(MemoryBudgetExceeded):
        with monitor:
            with pytest.raises(MemoryBudgetExceeded):
                monitor.checkpoint()
            with pytest.raises(MemoryBudgetExceeded):
                monitor.checkpoint()
    assert monitor.peak_rss_mib == 2.0
    with pytest.raises(MemoryBudgetExceeded):
        monitor.ensure_success()


def test_background_failure_interrupt_and_shutdown() -> None:
    interrupted = threading.Event()
    samples = iter([0, 2 * MIB])
    monitor = MemoryMonitor(
        1,
        interval_seconds=0.001,
        sampler=lambda: next(samples),
        interrupt=interrupted.set,
    )
    with pytest.raises(MemoryBudgetExceeded):
        with monitor:
            assert interrupted.wait(timeout=2)
    assert monitor.finished
    assert not any(thread.name == "crf-rss-monitor" for thread in threading.enumerate())


@pytest.mark.parametrize("observed", [-1, True, 0.5])
def test_invalid_sample_fails_closed(observed: int) -> None:
    monitor = MemoryMonitor(1, sampler=lambda: observed)
    with pytest.raises(ResourceMonitoringError, match="nonnegative integer"):
        with monitor:
            pass
    with pytest.raises(ResourceMonitoringError):
        monitor.ensure_success()


def test_monitoring_exception_and_body_abort() -> None:
    def denied() -> int:
        raise PermissionError("denied")

    with pytest.raises(ResourceMonitoringError, match="denied"):
        with MemoryMonitor(1, sampler=denied):
            pass
    monitor = MemoryMonitor(1, interval_seconds=60, sampler=lambda: 0)
    with pytest.raises(ValueError, match="body failed"):
        with monitor:
            raise ValueError("body failed")
    assert monitor.finished
    with pytest.raises(ResourceMonitoringError, match="unsuccessfully"):
        monitor.ensure_success()


def test_exit_sample_violation() -> None:
    samples = iter([0, 2 * MIB])
    monitor = MemoryMonitor(1, interval_seconds=60, sampler=lambda: next(samples))
    with pytest.raises(MemoryBudgetExceeded):
        with monitor:
            pass
    assert monitor.finished


def test_real_process_monitor_smoke() -> None:
    # Reads RSS only, without allocating memory or starting workers.
    with MemoryMonitor(1024 * 1024, interval_seconds=60) as monitor:
        monitor.checkpoint()
        assert monitor.peak_rss_mib > 0
    monitor.ensure_success()


def test_aggregate_rss_workers_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        def __init__(self, pid: int, rss: int, *, exited: bool = False) -> None:
            self.pid, self.rss, self.exited = pid, rss, exited

        def memory_info(self) -> Any:
            if self.exited:
                raise psutil.NoSuchProcess(self.pid)
            return SimpleNamespace(rss=self.rss)

        def children(self, recursive: bool = False) -> list[Process]:
            assert recursive
            return [child, child, grandchild, exited]

    root = Process(1, 100)
    child, grandchild, exited = (
        Process(2, 200),
        Process(3, 300),
        Process(4, 400, exited=True),
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: root)
    assert aggregate_rss_bytes() == 600
    root.exited = True
    with pytest.raises(ResourceMonitoringError):
        aggregate_rss_bytes()

    def denied(pid: int) -> Any:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    with pytest.raises(ResourceMonitoringError):
        aggregate_rss_bytes()


def test_logging_stderr_repeat_and_root_isolation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    handlers = root.handlers[:]
    logger = logging.getLogger("crf")
    old_handlers, old_level, old_propagate = (
        logger.handlers[:],
        logger.level,
        logger.propagate,
    )
    try:
        configure_logging()
        configure_logging(logging.DEBUG)
        logging.getLogger("crf.models.test").warning("diagnostic")
        print('{"result":1}')
        captured = capsys.readouterr()
        assert captured.out == '{"result":1}\n'
        assert captured.err.count("diagnostic") == 1
        assert "WARNING crf.models.test" in captured.err
        assert root.handlers == handlers
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


@pytest.mark.integration
def test_background_interrupt_stops_owned_pipeline() -> None:
    """Requires locked psutil and Python; interrupt only an isolated child process."""
    import subprocess
    import sys

    code = """
import time
from crf.utils.resources import MemoryMonitor, MemoryBudgetExceeded
samples = iter([0, 2 * 1024 * 1024])
try:
    with MemoryMonitor(1, interval_seconds=0.01, sampler=lambda: next(samples)):
        while True:
            time.sleep(0.01)
except MemoryBudgetExceeded:
    print("budget failure")
else:
    raise AssertionError("budget violation was ignored")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=5,
        check=True,
    )
    assert result.stdout == "budget failure\n"
    assert "exceeds" in result.stderr
