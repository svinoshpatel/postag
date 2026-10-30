"""Offline literature preprocessing: original inputs exist only in tmp_path."""

import hashlib
import os
import tracemalloc
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from crf.data.identity import (
    InputIdentity,
    discover_literature_paths,
    iter_literature_identities,
)
from crf.data.literature import (
    RawExclusion,
    RawSentence,
    _paragraphs,
    iter_literature,
    iter_literature_events,
    sentence_batches,
)
from crf.utils.config import PipelineConfig
from crf.utils.serialization import ValidationError


def source_file(root: Path, config: PipelineConfig, text: str | bytes) -> InputIdentity:
    data = text if isinstance(text, bytes) else text.encode(config.data.source_encoding)
    relative = "data/raw/literature/input.txt"
    (root / relative).write_bytes(data)
    return InputIdentity(
        1,
        "literature",
        relative,
        hashlib.sha256(data).hexdigest(),
        len(data),
        config.data.source_encoding,
    )


def events(
    root: Path, config: PipelineConfig, text: str | bytes
) -> list[RawSentence | RawExclusion]:
    return list(
        iter_literature_events(
            source_file(root, config, text), config=config, repository_root=root
        )
    )


@pytest.mark.parametrize("buffer", [1, 2, 3, 4, 7, 64, 65536])
@pytest.mark.parametrize("ending", ["\n", "\r", "\r\n"])
def test_buffer_independent_paragraphs_alignment(
    artifact_repository: Path, artifact_config: PipelineConfig, buffer: int, ending: str
) -> None:
    config = replace(
        artifact_config,
        resources=replace(artifact_config.resources, read_buffer_bytes=buffer),
    )
    text = f"\ufeffІ\u0308ї сім’я{ending}читає. Далі!{ending} \t{ending}П’ять.{ending}"
    result = events(artifact_repository, config, text)
    assert all(isinstance(item, RawSentence) for item in result)
    sentences = [item for item in result if isinstance(item, RawSentence)]
    assert [s.text.surface_text for s in sentences] == [
        f"І\u0308ї сім’я{ending}читає.",
        "Далі!",
        "П’ять.",
    ]
    assert [s.feature_forms for s in sentences] == [
        ("Її", "сім'я", "читає", "."),
        ("Далі", "!"),
        ("П'ять", "."),
    ]
    assert [s.paragraph_ordinal for s in sentences] == [1, 1, 2]
    assert [s.sentence_ordinal for s in sentences] == [1, 2, 3]
    for sentence in sentences:
        left, right = sentence.text.source_span
        assert text[left:right] == sentence.text.surface_text
        for form, (left, right) in zip(
            sentence.text.original_forms, sentence.text.token_spans, strict=True
        ):
            assert text[left:right] == form
        assert sentence.preprocessing.tokenizer_package_version == "2.0.0"
        assert sentence.preprocessing.tokenizer_adapter_version == 1
    with pytest.raises(FrozenInstanceError):
        sentences[0].paragraph_ordinal = 9  # type: ignore[misc]


@pytest.mark.parametrize("text", ["", " \t\n\r\n", "\ufeff", "\ufeff\n\n", "\n\n \t"])
def test_empty_inputs(
    artifact_repository: Path, artifact_config: PipelineConfig, text: str
) -> None:
    assert events(artifact_repository, artifact_config, text) == []


def test_crlf_and_whitespace_paragraph_lengths() -> None:
    paragraphs = list(_paragraphs(["  Кіт\r", "\nспить\r", "\n\t\r\n\r\nПес\r"], 100))
    assert [(p.start, p.length, p.ordinal, p.text) for p in paragraphs] == [
        (0, 12, 1, "  Кіт\r\nспить"),
        (19, 3, 2, "Пес"),
    ]


@pytest.mark.parametrize("encoding", ["cp1251", "utf-16", "utf-8-sig"])
def test_explicit_source_encoding(
    artifact_repository: Path, artifact_config: PipelineConfig, encoding: str
) -> None:
    config = replace(
        artifact_config,
        data=replace(artifact_config.data, source_encoding=encoding),
        resources=replace(artifact_config.resources, read_buffer_bytes=1),
    )
    result = events(artifact_repository, config, "Кіт читає.\n\nПес!")
    assert [s.feature_forms for s in result if isinstance(s, RawSentence)] == [
        ("Кіт", "читає", "."),
        ("Пес", "!"),
    ]
    assert isinstance(result[-1], RawSentence)
    assert result[-1].text.source_span == (12, 16)


@pytest.mark.parametrize("data", [b"\xff", b"\xd0", "Кіт.\n\n".encode() + b"\xd0"])
def test_bad_encoding_is_fatal(
    artifact_repository: Path, artifact_config: PipelineConfig, data: bytes
) -> None:
    config = replace(
        artifact_config,
        resources=replace(artifact_config.resources, read_buffer_bytes=1),
    )
    with pytest.raises(
        ValidationError, match=r"input.txt: utf-8 decoding error at byte"
    ):
        events(artifact_repository, config, data)


def test_discovery_never_opens_gold(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lit = artifact_repository / "data/raw/literature"
    (lit / "nested.txt").mkdir()
    (lit / "nested.txt/hidden.txt").write_text("Невидимий.", encoding="utf-8")
    (lit / "ignore.TXT").write_text("Ні.", encoding="utf-8")
    (lit / "ignore.conllu").write_bytes(b"not text")
    original = Path.open
    opened: list[Path] = []

    def guarded(self: Path, *args: object, **kwargs: object) -> object:
        assert "gold" not in self.parts
        opened.append(self)
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", guarded)
    sources = tuple(iter_literature_identities(artifact_config, artifact_repository))
    assert [Path(s.source_path).name for s in sources] == ["а.txt", "б.txt"]
    result = list(iter_literature(artifact_config, artifact_repository))
    assert len(result) == 2
    assert opened


@pytest.mark.parametrize(
    "kind",
    [
        "candidate_symlink",
        "dangling_symlink",
        "directory_symlink",
        "gold_hardlink",
        "missing",
        "empty",
    ],
)
def test_unsafe_or_empty_discovery(
    artifact_repository: Path, artifact_config: PipelineConfig, kind: str
) -> None:
    lit = artifact_repository / "data/raw/literature"
    if kind == "candidate_symlink":
        (lit / "bad.txt").symlink_to(artifact_repository / "data/raw/gold/train.conllu")
    elif kind == "dangling_symlink":
        (lit / "bad.txt").symlink_to(lit / "missing")
    elif kind == "gold_hardlink":
        os.link(artifact_repository / "data/raw/gold/train.conllu", lit / "bad.txt")
    else:
        for child in lit.iterdir():
            child.unlink()
        if kind == "directory_symlink":
            lit.rmdir()
            lit.symlink_to(
                artifact_repository / "data/raw/gold", target_is_directory=True
            )
        elif kind == "missing":
            lit.rmdir()
    with pytest.raises(ValueError):
        discover_literature_paths(artifact_config, artifact_repository)


@pytest.mark.parametrize(
    "text,limit,reason",
    [
        ("Кіт", 3, None),
        ("Кіт!", 3, "paragraph_chars"),
        ("Кіт\nПес", 7, None),
        ("Кіт\r\nПес", 7, "paragraph_chars"),
    ],
)
def test_paragraph_limit(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    text: str,
    limit: int,
    reason: str | None,
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources, max_paragraph_chars=limit, read_buffer_bytes=1
        ),
    )
    result = events(artifact_repository, config, text)
    assert len(result) == 1
    if reason:
        assert isinstance(result[0], RawExclusion)
        assert result[0].reason == reason
        assert result[0].characters == len(text)
    else:
        assert isinstance(result[0], RawSentence)


@pytest.mark.parametrize("limit,reason", [(4, None), (3, "sentence_chars")])
def test_sentence_limit_before_nfc(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    limit: int,
    reason: str | None,
) -> None:
    config = replace(
        artifact_config,
        resources=replace(artifact_config.resources, max_sentence_chars=limit),
    )
    result = events(artifact_repository, config, "І\u0308ї.")
    assert len(result) == 1
    if reason:
        assert isinstance(result[0], RawExclusion)
        assert result[0].reason == reason
        assert result[0].characters == 4
    else:
        assert isinstance(result[0], RawSentence)
        assert result[0].feature_forms == ("Її", ".")


@pytest.mark.parametrize("limit,excluded", [(4, False), (3, True)])
def test_token_limit_after_ud_adaptation(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    limit: int,
    excluded: bool,
) -> None:
    config = replace(
        artifact_config,
        resources=replace(artifact_config.resources, max_sentence_tokens=limit),
    )
    result = events(artifact_repository, config, "синьо-жовтий.")
    assert len(result) == 1
    if excluded:
        assert isinstance(result[0], RawExclusion)
        assert result[0].reason == "sentence_tokens"
        assert result[0].tokens == 4
    else:
        assert isinstance(result[0], RawSentence)


def test_oversized_recovery_and_ordinals(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources,
            max_paragraph_chars=20,
            max_sentence_chars=5,
            max_sentence_tokens=2,
            read_buffer_bytes=1,
        ),
    )
    text = "а" * 50 + "\nрядок\n\nДовгеньке. Кіт.\n\nА Б!\n\nПес!"
    result = events(artifact_repository, config, text)
    assert [s.reason for s in result if isinstance(s, RawExclusion)] == [
        "paragraph_chars",
        "sentence_chars",
        "sentence_tokens",
    ]
    assert [s.feature_forms for s in result if isinstance(s, RawSentence)] == [
        ("Кіт", "."),
        ("Пес", "!"),
    ]
    assert [s.paragraph_ordinal for s in result] == [1, 2, 2, 3, 4]
    assert [s.sentence_ordinal for s in result] == [None, 1, 2, 3, 4]
    assert "excluded paragraph_chars (56 characters)" in caplog.text
    assert isinstance(result[0], RawExclusion)
    assert result[0].source_span == (0, 56)


def test_draining_memory_is_bounded() -> None:
    def chunks() -> Iterator[str]:
        for _ in range(2000):
            yield "а" * 100
        yield "\n\nКіт"

    tracemalloc.start()
    try:
        result = list(_paragraphs(chunks(), 32))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 200_000
    assert result[0].text is None
    assert result[0].length == 200_000
    assert result[1].text == "Кіт"


def test_identity_failures_before_and_after_yield(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    source = source_file(artifact_repository, artifact_config, "Кіт.\n\nПес.")
    with pytest.raises(ValidationError, match="checksum/size mismatch"):
        list(
            iter_literature_events(
                replace(source, source_sha256="0" * 64),
                config=artifact_config,
                repository_root=artifact_repository,
            )
        )
    iterator = iter_literature_events(
        source, config=artifact_config, repository_root=artifact_repository
    )
    assert isinstance(next(iterator), RawSentence)
    (artifact_repository / source.source_path).write_text(
        "Пес.\n\nКіт.", encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="checksum/size mismatch"):
        list(iterator)
    with pytest.raises(ValidationError, match="literature source"):
        list(
            iter_literature_events(
                replace(source, source_role="test_gold"),
                config=artifact_config,
                repository_root=artifact_repository,
            )
        )


def test_file_boundaries_and_batches(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    lit = artifact_repository / "data/raw/literature"
    (lit / "а.txt").write_text("Кіт", encoding="utf-8")
    (lit / "б.txt").write_text("спить. Пес!", encoding="utf-8")
    sentences = [
        s
        for s in iter_literature(artifact_config, artifact_repository)
        if isinstance(s, RawSentence)
    ]
    assert [s.feature_forms for s in sentences] == [
        ("Кіт",),
        ("спить", "."),
        ("Пес", "!"),
    ]
    assert [s.sentence_ordinal for s in sentences] == [1, 1, 2]
    assert list(sentence_batches(sentences, token_limit=3)) == [
        (sentences[0], sentences[1]),
        (sentences[2],),
    ]
    assert list(sentence_batches([], token_limit=1)) == []
    with pytest.raises(ValidationError, match="cannot fit"):
        list(sentence_batches(sentences, token_limit=1))
    with pytest.raises(ValidationError):
        list(sentence_batches([], token_limit=True))


@pytest.mark.parametrize(
    "field,value",
    [
        ("read_buffer_bytes", 0),
        ("max_paragraph_chars", True),
        ("max_sentence_chars", -1),
        ("max_sentence_tokens", 5000),
        ("batch_tokens", 0),
    ],
)
def test_invalid_direct_resource_settings(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    field: str,
    value: object,
) -> None:
    config = replace(
        artifact_config, resources=replace(artifact_config.resources, **{field: value})
    )
    with pytest.raises(ValidationError):
        events(artifact_repository, config, "Кіт.")


def test_decode_failure_in_discarded_paragraph(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources, max_paragraph_chars=4, read_buffer_bytes=1
        ),
    )
    with pytest.raises(ValidationError, match="decoding error at byte 100"):
        events(artifact_repository, config, b"a" * 100 + b"\xff\n\nCat.")


def test_bounded_reads_and_incremental_yield(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = source_file(artifact_repository, artifact_config, "Кіт.\n\n" + "а" * 1000)
    config = replace(
        artifact_config,
        resources=replace(artifact_config.resources, read_buffer_bytes=3),
    )
    reads: list[int] = []
    original_open = Path.open

    class ObservedStream:
        def __init__(self, path: Path) -> None:
            self.stream = original_open(path, "rb")

        def __enter__(self) -> ObservedStream:
            return self

        def __exit__(self, *args: object) -> None:
            self.stream.close()

        def fileno(self) -> int:
            return self.stream.fileno()

        def read(self, size: int = -1) -> bytes:
            assert size == 3
            reads.append(size)
            return self.stream.read(size)

    def observed(self: Path, *args: object, **kwargs: object) -> object:
        if self != artifact_repository / source.source_path:
            return original_open(self, *args, **kwargs)  # type: ignore[arg-type]
        assert args == ("rb",)
        return ObservedStream(self)

    monkeypatch.setattr(Path, "open", observed)
    iterator = iter_literature_events(
        source, config=config, repository_root=artifact_repository
    )
    first = next(iterator)
    assert isinstance(first, RawSentence)
    initial_reads = len(reads)
    # One complete preflight hash, followed by only the first paragraph's bytes.
    assert initial_reads < 2 * ((source.size_bytes + 2) // 3)
    list(iterator)
    assert len(reads) > initial_reads
