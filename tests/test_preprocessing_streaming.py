"""Offline M4 integration: locked tokenizer/SQLite plus controlled RSS samplers.

No network, production corpora, fitted models or real low-memory runs. Temporary
JSONL below is a label-free diagnostic intermediate, not an M6 training dataset.
The test owner composes existing atomic publication and monitoring primitives.
"""

import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Self

import pytest

from crf.data.conllu import iter_gold_audit, iter_gold_records
from crf.data.contamination import (
    AuditReport,
    EligibleSentence,
    contamination_audit,
)
from crf.data.identity import (
    InputIdentity,
    digest_file,
    iter_input_identities,
    iter_literature_identities,
    verify_input_identity,
)
from crf.data.literature import (
    RawExclusion,
    RawSentence,
    iter_literature,
    iter_literature_events,
    sentence_batches,
)
from crf.utils.artifacts import atomic_writer
from crf.utils.config import PipelineConfig
from crf.utils.resources import (
    MemoryBudgetExceeded,
    MemoryMonitor,
    ResourceMonitoringError,
    check_memory,
)
from crf.utils.serialization import ValidationError, jsonl_bytes, loads_json

pytestmark = pytest.mark.integration
MIB = 1024 * 1024


@dataclass
class ControlledRSS:
    """Mutable deterministic sampler; tests change observations at stage boundaries."""

    rss: int = MIB
    failure: Exception | None = None
    calls: int = 0

    def __call__(self) -> int:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.rss

    def monitor(self) -> MemoryMonitor:
        return MemoryMonitor(1, interval_seconds=60, sampler=self)


@pytest.fixture
def streaming_corpus(artifact_repository: Path) -> Path:
    root = artifact_repository
    words = "".join(
        f"{index}\tслово{index}\t_\tNOUN\t_\t_\t0\troot\t_\t_\n"
        for index in range(1, 25)
    )
    for split in ("train", "val", "test"):
        (root / f"data/raw/gold/{split}.conllu").write_text(
            f"# text = Золотий {split}.\n" + words + "\n", encoding="utf-8"
        )
    for child in (root / "data/raw/literature").iterdir():
        child.unlink()
    (root / "data/raw/literature/a.txt").write_text(
        "Кіт спить.\n\nПес читає.\n\nПтах летить.", encoding="utf-8"
    )
    return root


@pytest.fixture
def streaming_config(artifact_config: PipelineConfig) -> PipelineConfig:
    return replace(
        artifact_config,
        resources=replace(
            artifact_config.resources,
            memory_budget_mib=1,
            read_buffer_bytes=7,
            max_paragraph_chars=48,
            max_sentence_chars=32,
            max_sentence_tokens=4,
            batch_tokens=6,
        ),
    )


def literature_source(root: Path, config: PipelineConfig) -> InputIdentity:
    return tuple(iter_literature_identities(config, root))[0]


def valid_lines(stream: BinaryIO) -> None:
    for line in stream:
        assert isinstance(loads_json(line), dict)


def write_intermediate(
    root: Path,
    config: PipelineConfig,
    monitor: MemoryMonitor,
    *,
    after_write: Callable[[int], None] | None = None,
    after_audit: Callable[[], None] | None = None,
) -> AuditReport:
    destination = root / "data/interim/run/eligible.jsonl"
    written = 0
    with atomic_writer(destination, root, validate=valid_lines) as stream:
        with monitor:
            with contamination_audit(
                config,
                root,
                index_path=Path("data/interim/run/contamination.sqlite3"),
                monitor=monitor,
            ) as audit:
                for event in audit.iter_events():
                    if not isinstance(event, EligibleSentence):
                        continue
                    before = stream.tell()
                    stream.write(
                        jsonl_bytes(
                            {
                                "sentence_id": event.sentence.sentence_id,
                                "fingerprint": event.sentence_fingerprint,
                                "forms": list(event.sentence.feature_forms),
                            }
                        )
                    )
                    stream.flush()
                    assert stream.tell() > before
                    assert not destination.exists()
                    written += 1
                    assert (
                        audit._connection.execute(
                            "SELECT COUNT(*) FROM seen"
                        ).fetchone()[0]
                        == written
                    )
                    if after_write is not None:
                        after_write(written)
                report = audit.report()
            if after_audit is not None:
                after_audit()
        monitor.ensure_success()
    return report


def instrument_source(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    state: ControlledRSS,
    *,
    fail_read: int | None = None,
    fail_line: int | None = None,
) -> list[int]:
    original_open = Path.open
    reads: list[int] = []
    lines = 0

    class SourceStream:
        def __init__(self) -> None:
            self.stream = original_open(target, "rb")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.stream.close()

        def fileno(self) -> int:
            return self.stream.fileno()

        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            result = self.stream.read(size)
            if len(reads) == fail_read:
                state.rss = MIB + 1
            return result

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> bytes:
            nonlocal lines
            raw = next(self.stream)
            lines += 1
            if lines == fail_line:
                state.rss = MIB + 1
            return raw

    def observed(self: Path, *args: object, **kwargs: object) -> object:
        if self == target and args == ("rb",):
            return SourceStream()
        return original_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", observed)
    return reads


def test_incremental_output_and_disk_indexes_at_exact_budget(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    state = ControlledRSS()
    monitor = state.monitor()
    report = write_intermediate(streaming_corpus, streaming_config, monitor)
    assert report.accepted_sentence_count == 3
    assert report.accepted_token_count == 9
    assert report.gold_sentence_count == 3
    assert monitor.peak_rss_mib == 1.0
    assert state.calls > 10
    output = streaming_corpus / "data/interim/run/eligible.jsonl"
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    assert not list(output.parent.glob(".partial-*"))
    monitor.ensure_success()


@pytest.mark.parametrize("failure", ["budget", "sampling", "final_exit"])
def test_resource_failure_never_publishes_partial_output(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    failure: str,
) -> None:
    raw = {
        path: path.read_bytes()
        for path in (streaming_corpus / "data/raw").rglob("*")
        if path.is_file()
    }
    state = ControlledRSS()
    monitor = state.monitor()
    writes: list[int] = []

    def after_write(count: int) -> None:
        writes.append(count)
        if failure == "budget":
            state.rss = MIB + 1
        elif failure == "sampling":
            state.failure = PermissionError("controlled RSS access failure")

    def after_audit() -> None:
        if failure == "final_exit":
            state.rss = MIB + 1

    expected = (
        ResourceMonitoringError if failure == "sampling" else MemoryBudgetExceeded
    )
    with pytest.raises(expected):
        write_intermediate(
            streaming_corpus,
            streaming_config,
            monitor,
            after_write=after_write,
            after_audit=after_audit,
        )
    assert writes
    assert monitor.finished
    with pytest.raises(expected):
        monitor.ensure_success()
    output = streaming_corpus / "data/interim/run/eligible.jsonl"
    assert not output.exists()
    assert not list(output.parent.glob(".partial-*"))
    assert {path: path.read_bytes() for path in raw} == raw


def test_hashing_checks_each_bounded_read(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = streaming_corpus / "data/raw/literature/a.txt"
    before = path.read_bytes()
    state = ControlledRSS()
    reads = instrument_source(monkeypatch, path, state, fail_read=3)
    monitor = state.monitor()
    with pytest.raises(MemoryBudgetExceeded), monitor:
        digest_file(path, buffer_bytes=7, monitor=monitor)
    assert reads == [7, 7, 7]
    state.rss = 0
    with pytest.raises(MemoryBudgetExceeded):
        monitor.ensure_success()
    assert path.read_bytes() == before


def test_raw_drain_fails_before_emitting_a_rejection_fragment(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crf.data.literature as literature

    path = streaming_corpus / "data/raw/literature/a.txt"
    path.write_text("а" * 200 + "\n\nКіт.", encoding="utf-8")
    config = replace(
        streaming_config,
        resources=replace(streaming_config.resources, max_paragraph_chars=16),
    )
    source = literature_source(streaming_corpus, config)
    state = ControlledRSS()
    original_units = literature._line_units
    observed: list[str] = []

    def units(chunks: Iterator[str]) -> Iterator[str]:
        for count, unit in enumerate(original_units(chunks), 1):
            if count == 20:
                state.rss = MIB + 1
            yield unit

    monkeypatch.setattr(literature, "_line_units", units)
    monitor = state.monitor()
    with pytest.raises(MemoryBudgetExceeded), monitor:
        for event in iter_literature_events(
            source, config=config, repository_root=streaming_corpus, monitor=monitor
        ):
            observed.append(str(event))
    assert observed == []
    assert path.read_text(encoding="utf-8") == "а" * 200 + "\n\nКіт."


@pytest.mark.parametrize("reader", ["audit", "records"])
def test_gold_resource_failure_is_fatal_without_partial_sentences(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
) -> None:
    source = tuple(iter_input_identities(streaming_config, streaming_corpus))[0]
    path = streaming_corpus / source.source_path
    before = path.read_bytes()
    state = ControlledRSS()
    instrument_source(monkeypatch, path, state, fail_line=5)
    monitor = state.monitor()
    emitted: list[object] = []
    with pytest.raises(MemoryBudgetExceeded), monitor:
        iterator = (
            iter_gold_audit(source, streaming_config, streaming_corpus, monitor=monitor)
            if reader == "audit"
            else iter_gold_records(
                source, streaming_config, streaming_corpus, monitor=monitor
            )
        )
        for sentence in iterator:
            emitted.append(sentence)
    assert emitted == []
    assert path.read_bytes() == before


def test_gold_retains_all_words_despite_raw_limits(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    state = ControlledRSS()
    monitor = state.monitor()
    source = tuple(iter_input_identities(streaming_config, streaming_corpus))[0]
    with monitor:
        records = list(
            iter_gold_records(
                source, streaming_config, streaming_corpus, monitor=monitor
            )
        )
    monitor.ensure_success()
    assert len(records) == 1
    assert records[0].token_ids == tuple(range(1, 25))
    assert len(records[0].feature_forms) == len(records[0].target_labels) == 24


def test_whole_raw_candidate_rejections_keep_spans_and_recover(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    text = "а" * 49 + "\n\n" + "б" * 33 + ".\n\nА Б В Г.\n\nКіт."
    (streaming_corpus / "data/raw/literature/a.txt").write_text(text, encoding="utf-8")
    monitor = ControlledRSS().monitor()
    with monitor:
        events = list(
            iter_literature(streaming_config, streaming_corpus, monitor=monitor)
        )
    assert [e.reason for e in events if isinstance(e, RawExclusion)] == [
        "paragraph_chars",
        "sentence_chars",
        "sentence_tokens",
    ]
    assert [e.characters for e in events if isinstance(e, RawExclusion)] == [49, 34, 8]
    assert [e.feature_forms for e in events if isinstance(e, RawSentence)] == [
        ("Кіт", ".")
    ]
    assert [e.sentence_ordinal for e in events] == [None, 1, 2, 3]
    for event in events:
        span = (
            event.source_span
            if isinstance(event, RawExclusion)
            else event.text.source_span
        )
        assert len(text[span[0] : span[1]]) == (
            event.characters
            if isinstance(event, RawExclusion)
            else len(event.text.surface_text)
        )
    monitor.ensure_success()


def test_inference_retains_sentences_removed_by_pseudo_policy(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    path = streaming_corpus / "data/raw/gold/train.conllu"
    path.write_text(
        "# text = Золотий.\n1-2\tаби-то\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tаби\t_\tSCONJ\t_\t_\t0\troot\t_\t_\n"
        "2\tто\t_\tPART\t_\t_\t1\tdep\t_\t_\n\n",
        encoding="utf-8",
    )
    (streaming_corpus / "data/raw/literature/a.txt").write_text(
        "Золотий.\n\nаби-то.\n\nКіт.\n\nКіт.", encoding="utf-8"
    )
    with ControlledRSS().monitor() as monitor:
        raw = list(iter_literature(streaming_config, streaming_corpus, monitor=monitor))
    assert len(raw) == 4
    assert all(isinstance(e, RawSentence) for e in raw)
    with ControlledRSS().monitor() as monitor:
        with contamination_audit(
            streaming_config,
            streaming_corpus,
            index_path=Path("data/interim/policy/index.sqlite3"),
            monitor=monitor,
        ) as audit:
            events = list(audit.iter_events())
            report = audit.report()
    assert len([e for e in events if isinstance(e, EligibleSentence)]) == 1
    assert {key: value for key, value in report.exclusion_counts if value} == {
        "gold_overlap": 1,
        "known_mwt": 1,
        "literature_duplicate": 1,
    }


def test_batches_have_one_candidate_lookahead_and_stop_on_budget_failure(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    sentences = [
        e
        for e in iter_literature(streaming_config, streaming_corpus)
        if isinstance(e, RawSentence)
    ]
    consumed = 0

    def candidates() -> Iterator[RawSentence]:
        nonlocal consumed
        for sentence in sentences:
            consumed += 1
            yield sentence

    state = ControlledRSS()
    monitor = state.monitor()
    with pytest.raises(MemoryBudgetExceeded), monitor:
        batches = sentence_batches(candidates(), token_limit=3, monitor=monitor)
        first = next(batches)
        assert consumed == 2
        assert sum(len(s.feature_forms) for s in first) == 3
        state.rss = MIB + 1
        next(batches)


@pytest.mark.parametrize("kind", ["inactive", "finished", "wrong_budget"])
def test_invalid_monitor_state_blocks_before_any_index_output(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    kind: str,
) -> None:
    state = ControlledRSS()
    monitor = (
        state.monitor()
        if kind != "wrong_budget"
        else MemoryMonitor(2, interval_seconds=60, sampler=state)
    )
    index = Path("data/interim/blocked/index.sqlite3")
    if kind == "finished":
        with monitor:
            pass
    if kind == "wrong_budget":
        with pytest.raises(ResourceMonitoringError, match="budget"), monitor:
            with contamination_audit(
                streaming_config, streaming_corpus, index_path=index, monitor=monitor
            ):
                pytest.fail("Wrong budget cannot create an audit")
    else:
        with pytest.raises(ResourceMonitoringError, match="not active"):
            with contamination_audit(
                streaming_config, streaming_corpus, index_path=index, monitor=monitor
            ):
                pytest.fail("Inactive monitor cannot create an audit")
    assert not (streaming_corpus / index).exists()


@pytest.mark.parametrize("failure", [False, True])
def test_report_checks_final_monitor_outcome(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    failure: bool,
) -> None:
    state = ControlledRSS()
    monitor = state.monitor()
    with ExitStack() as stack:
        try:
            with monitor:
                audit = stack.enter_context(
                    contamination_audit(
                        streaming_config,
                        streaming_corpus,
                        index_path=Path("data/interim/report/index.sqlite3"),
                        monitor=monitor,
                    )
                )
                list(audit.iter_events())
                assert audit.report().accepted_sentence_count == 3
                if failure:
                    state.rss = MIB + 1
        except MemoryBudgetExceeded:
            assert failure
        if failure:
            with pytest.raises(MemoryBudgetExceeded):
                audit.report()
        else:
            assert audit.report().accepted_sentence_count == 3
            monitor.ensure_success()


def test_large_stream_does_not_retain_prior_sentence_objects(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crf.data.contamination as contamination

    path = streaming_corpus / "data/raw/literature/a.txt"
    with path.open("w", encoding="utf-8") as stream:
        for ordinal in range(300):
            stream.write(f"Речення {ordinal}.\n\n")
    live: weakref.WeakSet[RawSentence] = weakref.WeakSet()
    peak = 0
    original = contamination.iter_literature_events

    def tracked(
        *args: object, **kwargs: object
    ) -> Iterator[RawSentence | RawExclusion]:
        nonlocal peak
        for event in original(*args, **kwargs):  # type: ignore[arg-type]
            if isinstance(event, RawSentence):
                live.add(event)
                peak = max(peak, len(live))
            yield event

    monkeypatch.setattr(contamination, "iter_literature_events", tracked)
    report = write_intermediate(
        streaming_corpus, streaming_config, ControlledRSS().monitor()
    )
    assert report.accepted_sentence_count == 300
    assert peak <= 4
    assert len(live) == 0


def test_check_memory_none_preserves_unmonitored_calls() -> None:
    check_memory(None, expected_budget_mib=1)


def test_identity_verification_samples_before_source_io(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = literature_source(streaming_corpus, streaming_config)
    state = ControlledRSS()
    monitor = state.monitor()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Failed monitor must stop before source access")

    with pytest.raises(MemoryBudgetExceeded), monitor:
        state.rss = MIB + 1
        monkeypatch.setattr(Path, "open", forbidden)
        verify_input_identity(
            source, streaming_config, streaming_corpus, monitor=monitor
        )


def test_partial_gold_index_failure_closes_database(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    import crf.data.contamination as contamination

    state = ControlledRSS()
    original_audit = contamination.iter_gold_audit
    original_connect = contamination.sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(connection)
        return connection

    def audits(*args: object, **kwargs: object) -> Iterator[object]:
        for sentence in original_audit(*args, **kwargs):  # type: ignore[arg-type]
            state.rss = MIB + 1
            yield sentence

    monkeypatch.setattr(contamination.sqlite3, "connect", connect)
    monkeypatch.setattr(contamination, "iter_gold_audit", audits)
    monitor = state.monitor()
    with pytest.raises(MemoryBudgetExceeded), monitor:
        with contamination_audit(
            streaming_config,
            streaming_corpus,
            index_path=Path("data/interim/failed/index.sqlite3"),
            monitor=monitor,
        ):
            pytest.fail("Failed gold audit must not yield a filter")
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert not list((streaming_corpus / "data/interim").rglob("*complete*"))


def test_early_termination_prevents_atomic_output_publication(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    destination = streaming_corpus / "data/interim/early/eligible.jsonl"
    monitor = ControlledRSS().monitor()
    with pytest.raises(ValidationError, match="incomplete"):
        with atomic_writer(
            destination, streaming_corpus, validate=valid_lines
        ) as stream:
            with monitor:
                with contamination_audit(
                    streaming_config,
                    streaming_corpus,
                    index_path=Path("data/interim/early/index.sqlite3"),
                    monitor=monitor,
                ) as audit:
                    iterator = audit.iter_events()
                    first = next(iterator)
                    assert isinstance(first, EligibleSentence)
                    stream.write(
                        jsonl_bytes({"fingerprint": first.sentence_fingerprint})
                    )
                    iterator.close()
                    audit.report()
            monitor.ensure_success()
    assert not destination.exists()
    assert not list(destination.parent.glob(".partial-*"))
    with pytest.raises(ResourceMonitoringError, match="unsuccessfully"):
        monitor.ensure_success()


def test_late_decoding_failure_cleans_streamed_output(
    streaming_corpus: Path,
    streaming_config: PipelineConfig,
) -> None:
    (streaming_corpus / "data/raw/literature/z.txt").write_bytes(b"\xff")
    writes: list[int] = []
    monitor = ControlledRSS().monitor()
    with pytest.raises(ValidationError, match="decoding error"):
        write_intermediate(
            streaming_corpus, streaming_config, monitor, after_write=writes.append
        )
    assert writes == [1, 2, 3]
    assert not (streaming_corpus / "data/interim/run/eligible.jsonl").exists()
    assert not list((streaming_corpus / "data/interim/run").glob(".partial-*"))
