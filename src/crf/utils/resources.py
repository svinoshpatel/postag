"""Best-effort aggregate RSS monitoring for an owned pipeline process."""

import _thread
import logging
import math
import os
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Literal

import psutil

logger = logging.getLogger(__name__)
_MIB = 1024 * 1024


class ResourceMonitoringError(RuntimeError):
    """Resource use cannot be monitored reliably; the run must fail."""


class MemoryBudgetExceeded(ResourceMonitoringError):
    """Observed main-plus-worker RSS exceeded the configured budget."""


def aggregate_rss_bytes() -> int:
    """Sum the current process and recursive workers, ignoring exited children."""
    try:
        root = psutil.Process(os.getpid())
        total = root.memory_info().rss
        seen = {root.pid}
        for child in root.children(recursive=True):
            if child.pid in seen:
                continue
            seen.add(child.pid)
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total
    except psutil.Error as exc:
        raise ResourceMonitoringError(f"Cannot sample process-tree RSS: {exc}") from exc


class MemoryMonitor:
    """Mutable controller: peak, first failure, and thread lifecycle evolve per run."""

    def __init__(
        self,
        memory_budget_mib: int,
        *,
        interval_seconds: float = 0.1,
        sampler: Callable[[], int] = aggregate_rss_bytes,
        interrupt: Callable[[], None] = _thread.interrupt_main,
    ) -> None:
        if type(memory_budget_mib) is not int or memory_budget_mib <= 0:
            raise ValueError("memory_budget_mib must be a positive integer")
        if (
            type(interval_seconds) not in (int, float)
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        self.budget_mib = memory_budget_mib
        self.interval_seconds = interval_seconds
        self._sampler = sampler
        self._interrupt = interrupt
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes = 0
        self._failure: ResourceMonitoringError | None = None
        self._started = False
        self._finished = False
        self._aborted = False

    @property
    def peak_rss_mib(self) -> float:
        with self._lock:
            return self._peak_bytes / _MIB

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def _sample(self) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            try:
                observed = self._sampler()
                if type(observed) is not int or observed < 0:
                    raise ResourceMonitoringError(
                        "RSS sampler must return nonnegative integer bytes"
                    )
                self._peak_bytes = max(self._peak_bytes, observed)
                if observed > self.budget_mib * _MIB:
                    raise MemoryBudgetExceeded(
                        f"Aggregate RSS {observed / _MIB:.3f} MiB exceeds {self.budget_mib} MiB budget"
                    )
            except Exception as exc:
                failure = (
                    exc
                    if isinstance(exc, ResourceMonitoringError)
                    else ResourceMonitoringError(f"RSS sampling failed: {exc}")
                )
                self._failure = failure
                logger.error("%s", failure)
                if failure is exc:
                    raise
                raise failure from exc

    def checkpoint(self) -> None:
        """Sample at a stage boundary; previously observed failure stays fatal."""
        if not self._started or self.finished:
            raise ResourceMonitoringError("Memory monitor is not active")
        self._sample()

    def ensure_success(self) -> None:
        """Authorize publication only after monitoring completed without failure."""
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._aborted:
                raise ResourceMonitoringError("Pipeline context exited unsuccessfully")
            if not self._finished:
                raise ResourceMonitoringError(
                    "Successful publication requires a finished memory monitor"
                )

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except ResourceMonitoringError:
                self._interrupt()
                return

    def __enter__(self) -> MemoryMonitor:
        if threading.current_thread() is not threading.main_thread():
            raise ResourceMonitoringError(
                "Memory monitoring must run on the pipeline main thread"
            )
        if self._started:
            raise ResourceMonitoringError("Memory monitor cannot be reused")
        self._started = True
        try:
            self._sample()
            self._thread = threading.Thread(
                target=self._watch, name="crf-rss-monitor", daemon=True
            )
            self._thread.start()
        except BaseException:
            self._aborted = True
            self._finished = True
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._aborted = exc_type is not None
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
            if self._failure is None:
                self._sample()
            if exc_type is not None and not issubclass(exc_type, KeyboardInterrupt):
                return False
            if self._failure is not None:
                raise self._failure from exc
        except KeyboardInterrupt as interrupt:
            self._aborted = True
            if self._failure is not None:
                raise self._failure from interrupt
            raise
        finally:
            self._finished = True
        return False
