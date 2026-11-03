"""Offline text-only contamination policy; SQLite/raw inputs live in tmp_path."""

import hashlib
import sqlite3
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from crf.data.contamination import (
    AuditReport,
    ContaminationExclusion,
    EligibleSentence,
    KnownSourceExclusion,
    contamination_audit,
    sentence_fingerprint,
)
from crf.data.identity import GOLD_ROLES, InputIdentity, iter_input_identities
from crf.data.literature import RawExclusion, RawSentence, iter_literature
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryBudgetExceeded, MemoryMonitor
from crf.utils.serialization import ValidationError, json_bytes, loads_json


def word(identifier: str, form: str, misc: str = "_") -> str:
    # Intentionally invalid annotations: auditing must leave them uninterpreted.
    return f"{identifier}\t{form}\tSECRET_LEMMA\tINVALID_UPOS\tSECRET_XPOS\tBAD_FEATS\tBAD_HEAD\tBAD_REL\tBAD_DEPS\t{misc}\n"


def gold(text: str, form: str = "Кіт") -> str:
    return f"# text = {text}\n" + word("1", form) + "\n"


def mwt(surface: str, text: str) -> str:
    return (
        f"# text = {text}\n"
        + word("1-2", surface)
        + word("1", "аби")
        + word("2", "то")
        + "\n"
    )


@pytest.fixture
def corpus(artifact_repository: Path) -> Path:
    root = artifact_repository
    for role in GOLD_ROLES:
        (root / f"data/raw/gold/{role.removesuffix('_gold')}.conllu").write_text(
            gold(
                {
                    "train_gold": "Навчальний.",
                    "val_gold": "Перевірочний.",
                    "test_gold": "Тестовий.",
                }[role]
            ),
            encoding="utf-8",
        )
    for child in (root / "data/raw/literature").iterdir():
        child.unlink()
    (root / "data/raw/literature/a.txt").write_text("Новий текст.", encoding="utf-8")
    return root


def index(name: str = "run") -> Path:
    return Path(f"data/interim/{name}/contamination.sqlite3")


def run(
    root: Path,
    config: PipelineConfig,
    name: str = "run",
    rules: tuple[KnownSourceExclusion, ...] = (),
) -> tuple[list[object], AuditReport]:
    with contamination_audit(
        config, root, index_path=index(name), known_source_exclusions=rules
    ) as audit:
        result = list(audit.iter_events())
        report = audit.report()
    return result, report


@pytest.mark.parametrize(
    "left,right",
    [
        ("  Кіт\u00a0\n спить. ", "Кіт спить."),
        ("Її сім’я.", "Її сім'я."),
        ("І\u0308ї пʼять.", "Її п`ять."),
        ("Кіт\u202f\tспить.", "Кіт спить."),
    ],
)
def test_fingerprint_exact_normalization(left: str, right: str) -> None:
    normalized = (
        "Кіт спить."
        if "Кіт" in right
        else "Її сім'я."
        if "сім" in right
        else "Її п'ять."
    )
    expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    assert sentence_fingerprint(left) == sentence_fingerprint(right) == expected


@pytest.mark.parametrize(
    "other", ["кіт.", "Кіт!", "Кіт", "Кі\u200bт.", "Кі́т.", "\ufeffКіт."]
)
def test_fingerprint_preserves_distinctions(other: str) -> None:
    assert sentence_fingerprint(other) != sentence_fingerprint("Кіт.")


@pytest.mark.parametrize("text", ["", " \n\t", "\u00a0\u202f"])
def test_empty_fingerprint_fails(text: str) -> None:
    with pytest.raises(ValueError):
        sentence_fingerprint(text)


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_unsupported_fingerprint_version(version: object) -> None:
    with pytest.raises(ValidationError):
        sentence_fingerprint("Кіт.", version=version)  # type: ignore[arg-type]


def test_all_gold_splits_are_excluded_without_annotation_access(
    corpus: Path, artifact_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "Навчальний. Перевірочний. Тестовий. Новий текст."
    (corpus / "data/raw/literature/a.txt").write_text(text, encoding="utf-8")
    before = {p: p.read_bytes() for p in (corpus / "data/raw/gold").iterdir()}

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Labeled gold interface must never be used")

    monkeypatch.setattr("crf.data.conllu.iter_gold_sentences", forbidden)
    monkeypatch.setattr("crf.data.conllu.iter_gold_records", forbidden)
    monkeypatch.setattr("crf.data.conllu.validate_upos", forbidden)
    result, report = run(corpus, artifact_config)
    excluded = [e for e in result if isinstance(e, ContaminationExclusion)]
    assert [e.reason for e in excluded] == ["gold_overlap"] * 3
    assert [e.matched_splits for e in excluded] == [
        ("train_gold",),
        ("val_gold",),
        ("test_gold",),
    ]
    assert [
        e.sentence.feature_forms for e in result if isinstance(e, EligibleSentence)
    ] == [("Новий", "текст", ".")]
    assert report.gold_sentence_count == 3
    assert report.accepted_sentence_count == 1
    assert dict(report.exclusion_counts)["gold_overlap"] == 3
    assert {p: p.read_bytes() for p in before} == before
    assert "SECRET" not in str(result) + str(report.to_dict())
    with pytest.raises(FrozenInstanceError):
        report.accepted_sentence_count = 5  # type: ignore[misc]


def test_gold_text_precedence_mwt_reconstruction_and_empty_nodes(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    train = corpus / "data/raw/gold/train.conllu"
    train.write_text(
        gold("Реальний текст.", "Інший")
        + word("1-2", "аби-то", "SpaceAfter=No")
        + word("1", "аби")
        + word("1.1", "ПРИВИД")
        + word("2", "то")
        + word("3", ".")
        + "\n",
        encoding="utf-8",
    )
    (corpus / "data/raw/literature/a.txt").write_text(
        "Реальний текст.\n\nаби-то.\n\nІнший.", encoding="utf-8"
    )
    result, report = run(corpus, artifact_config)
    assert [e.reason for e in result if isinstance(e, ContaminationExclusion)] == [
        "gold_overlap",
        "gold_overlap",
    ]
    assert [
        e.sentence.text.surface_text for e in result if isinstance(e, EligibleSentence)
    ] == ["Інший."]
    assert report.mwt_pattern_count == 1


def test_same_text_in_multiple_splits(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    for path in (corpus / "data/raw/gold").iterdir():
        path.write_text(gold("Кіт."), encoding="utf-8")
    (corpus / "data/raw/literature/a.txt").write_text("Кіт.", encoding="utf-8")
    result, report = run(corpus, artifact_config)
    assert isinstance(result[0], ContaminationExclusion)
    assert result[0].matched_splits == ("test_gold", "train_gold", "val_gold")
    assert report.gold_sentence_count == 3
    assert report.accepted_sentence_count == 0


def test_dedup_discovery_order_and_whitespace_normalization(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    lit = corpus / "data/raw/literature"
    (lit / "z.txt").write_text("Кіт спить. КІТ спить.", encoding="utf-8")
    (lit / "a.txt").write_text("Кіт\nспить. Кіт\u00a0спить.", encoding="utf-8")
    result, report = run(corpus, artifact_config)
    eligible = [e for e in result if isinstance(e, EligibleSentence)]
    duplicate = [e for e in result if isinstance(e, ContaminationExclusion)]
    assert len(eligible) == 2
    assert [Path(e.sentence.source.source_path).name for e in eligible] == [
        "a.txt",
        "z.txt",
    ]
    assert [e.duplicate_of for e in duplicate] == [eligible[0].sentence.sentence_id] * 2
    assert [e.reason for e in duplicate] == ["literature_duplicate"] * 2
    assert report.accepted_token_count == 6
    assert dict(report.exclusion_counts)["literature_duplicate"] == 2
    # A fresh index and identical sources reproduce the entire automatic report.
    again, repeated = run(corpus, artifact_config, "repeat")
    assert again == result
    assert repeated == report
    assert loads_json(json_bytes(report.to_dict())) == report.to_dict()


def test_train_mwt_patterns_are_boundary_aware_and_heldout_patterns_ignored(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/gold/train.conllu").write_text(
        mwt("аби-то", "Інше речення."), encoding="utf-8"
    )
    (corpus / "data/raw/gold/test.conllu").write_text(
        mwt("хтозна", "Тестовий."), encoding="utf-8"
    )
    (corpus / "data/raw/gold/val.conllu").write_text(
        mwt("невже", "Перевірочний."), encoding="utf-8"
    )
    (corpus / "data/raw/literature/a.txt").write_text(
        "Це аби-то.\n\nЦе аби - то.\n\nЦе преаби-то.\n\nЦе Аби-то.\n\nхтозна невже.",
        encoding="utf-8",
    )
    result, report = run(corpus, artifact_config)
    rejected = [e for e in result if isinstance(e, ContaminationExclusion)]
    assert [e.reason for e in rejected] == ["known_mwt", "known_mwt"]
    assert [e.matched_mwt for e in rejected] == ["аби-то", "аби-то"]
    assert report.mwt_pattern_count == 1
    assert report.accepted_sentence_count == 3


def test_multi_token_mwt_surface_is_matched(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/gold/train.conllu").write_text(
        mwt("наче б", "Навчальний."), encoding="utf-8"
    )
    (corpus / "data/raw/literature/a.txt").write_text(
        "Це наче б. Це неначе б.", encoding="utf-8"
    )
    result, report = run(corpus, artifact_config)
    assert isinstance(result[0], ContaminationExclusion)
    assert result[0].matched_mwt == "наче б"
    assert isinstance(result[1], EligibleSentence)


@pytest.mark.parametrize(
    "surfaces,text,expected",
    [
        (("наче", "наче б"), "Це наче б.", "наче б"),
        (("наче б", "наче"), "Це наче б.", "наче б"),
        (("наче", "аби то так"), "наче аби то так.", "наче"),
    ],
)
def test_mwt_match_preserves_longest_prefix_at_earliest_position(
    corpus: Path,
    artifact_config: PipelineConfig,
    surfaces: tuple[str, ...],
    text: str,
    expected: str,
) -> None:
    (corpus / "data/raw/gold/train.conllu").write_text(
        "".join(mwt(surface, "Навчальний.") for surface in surfaces),
        encoding="utf-8",
    )
    (corpus / "data/raw/literature/a.txt").write_text(text, encoding="utf-8")
    result, report = run(corpus, artifact_config)
    assert len(result) == 1
    assert isinstance(result[0], ContaminationExclusion)
    assert result[0].reason == "known_mwt"
    assert result[0].matched_mwt == expected
    assert report.accepted_sentence_count == 0
    assert dict(report.exclusion_counts)["known_mwt"] == 1


def test_mwt_matching_serializes_once_per_distinct_length(
    corpus: Path, artifact_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Many train patterns share the same head/length but none matches the raw text.
    surfaces = [
        f"наче варіант{chr(0x430 + number // 32)}{chr(0x430 + number % 32)}"
        for number in range(80)
    ]
    (corpus / "data/raw/gold/train.conllu").write_text(
        "".join(mwt(surface, "Навчальний.") for surface in surfaces),
        encoding="utf-8",
    )
    (corpus / "data/raw/literature/a.txt").write_text("наче інший.", encoding="utf-8")
    sentence = next(iter_literature(artifact_config, corpus))
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        encoded: list[object] = []

        def observe_encoding(value: object) -> bytes:
            encoded.append(value)
            return json_bytes(value)

        monkeypatch.setattr("crf.data.contamination.json_bytes", observe_encoding)
        assert audit._match_mwt(sentence) is None
        assert encoded == [["наче", "інший"]]


def test_memory_failure_during_mwt_probe_prevents_report(
    corpus: Path, artifact_config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    (corpus / "data/raw/gold/train.conllu").write_text(
        mwt("наче б", "Навчальний."), encoding="utf-8"
    )
    (corpus / "data/raw/literature/a.txt").write_text("наче б.", encoding="utf-8")
    armed = False
    samples = 0

    def sample() -> int:
        nonlocal samples
        if armed:
            samples += 1
        return (
            artifact_config.resources.memory_budget_mib * 1024 * 1024 + 1
            if samples >= 2
            else 0
        )

    monitor = MemoryMonitor(
        artifact_config.resources.memory_budget_mib,
        interval_seconds=60,
        sampler=sample,
    )
    with pytest.raises(MemoryBudgetExceeded), monitor:
        with contamination_audit(
            artifact_config, corpus, index_path=index(), monitor=monitor
        ) as audit:
            match = audit._match_mwt

            def arm_matching(sentence: RawSentence) -> str | None:
                nonlocal armed
                armed = True
                return match(sentence)

            monkeypatch.setattr(audit, "_match_mwt", arm_matching)
            statements: list[str] = []
            audit._connection.set_trace_callback(statements.append)
            with pytest.raises(MemoryBudgetExceeded):
                list(audit.iter_events())
            assert samples == 2
            assert not any("SELECT surface FROM mwt" in sql for sql in statements)
            with pytest.raises(ValidationError, match="incomplete"):
                audit.report()
    with pytest.raises(MemoryBudgetExceeded):
        monitor.ensure_success()


def test_known_source_skipped_in_full_before_decoding_or_dedup(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    lit = corpus / "data/raw/literature"
    (lit / "a.txt").write_bytes(b"\xffnot valid UTF-8")
    (lit / "b.txt").write_text("Новий текст.", encoding="utf-8")
    rule = KnownSourceExclusion(
        "data/raw/literature/a.txt",
        "test_gold",
        "Same documented edition as held-out source",
    )
    result, report = run(corpus, artifact_config, rules=(rule,))
    assert len(result) == 2
    assert isinstance(result[0], ContaminationExclusion)
    assert result[0].reason == "known_source"
    assert result[0].sentence_id is result[0].characters is None
    assert result[0].evidence == rule.evidence
    assert isinstance(result[1], EligibleSentence)
    assert report.excluded_source_bytes == len(b"\xffnot valid UTF-8")
    assert report.known_source_exclusions == (rule,)
    assert dict(report.exclusion_counts)["known_source"] == 1


@pytest.mark.parametrize(
    "rule",
    [
        KnownSourceExclusion(
            "data/raw/literature/missing.txt", "test_gold", "evidence"
        ),
        KnownSourceExclusion("data/raw/gold/test.conllu", "test_gold", "evidence"),
        KnownSourceExclusion("../bad.txt", "test_gold", "evidence"),
        KnownSourceExclusion("data/raw/literature/a.txt", "train_gold", "evidence"),  # type: ignore[arg-type]
        KnownSourceExclusion("data/raw/literature/a.txt", "test_gold", " \t"),
    ],
)
def test_bad_known_source_rules_fail_before_index_creation(
    corpus: Path, artifact_config: PipelineConfig, rule: KnownSourceExclusion
) -> None:
    with (
        pytest.raises(ValidationError),
        contamination_audit(
            artifact_config, corpus, index_path=index(), known_source_exclusions=(rule,)
        ),
    ):
        pytest.fail("Must reject invalid rules")
    assert not (corpus / index()).exists()


def test_duplicate_rule_rejected(corpus: Path, artifact_config: PipelineConfig) -> None:
    rule = KnownSourceExclusion("data/raw/literature/a.txt", "val_gold", "evidence")
    with (
        pytest.raises(ValidationError, match="duplicate"),
        contamination_audit(
            artifact_config,
            corpus,
            index_path=index(),
            known_source_exclusions=(rule, rule),
        ),
    ):
        pass


def test_precedence_and_resource_events_do_not_change_gold(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources,
            max_sentence_tokens=2,
            max_sentence_chars=20,
            max_paragraph_chars=60,
        ),
    )
    # The same surface is one token when digit-group whitespace is ordinary space;
    # the newline version exceeds the token limit and must not reserve a slot.
    (corpus / "data/raw/gold/train.conllu").write_text(
        mwt("аби-то", "Навчальний."), encoding="utf-8"
    )
    text = "1\n000\n000.\n\n1 000 000.\n\n" + "а" * 80 + "\n\nКіт."
    (corpus / "data/raw/literature/a.txt").write_text(text, encoding="utf-8")
    before = (corpus / "data/raw/gold/train.conllu").read_bytes()
    result, report = run(corpus, config)
    assert isinstance(result[0], RawExclusion)
    assert result[0].reason == "sentence_tokens"
    assert isinstance(result[1], EligibleSentence)
    assert result[1].sentence.feature_forms == ("1 000 000", ".")
    assert isinstance(result[2], RawExclusion)
    assert result[2].reason == "paragraph_chars"
    assert report.accepted_sentence_count == 2
    assert dict(report.exclusion_counts)["literature_duplicate"] == 0
    assert (corpus / "data/raw/gold/train.conllu").read_bytes() == before


def test_report_requires_exhaustion_and_audit_is_single_use(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        with pytest.raises(ValidationError, match="incomplete"):
            audit.report()
        iterator = audit.iter_events()
        assert isinstance(next(iterator), EligibleSentence)
        with pytest.raises(ValidationError, match="incomplete"):
            audit.report()
        iterator.close()
        with pytest.raises(ValidationError, match="consumed"):
            list(audit.iter_events())
    with pytest.raises(ValidationError, match="closed"):
        audit.report()


@pytest.mark.parametrize("change", ["gold", "literature", "added", "removed"])
def test_input_changes_prevent_completion(
    corpus: Path, artifact_config: PipelineConfig, change: str
) -> None:
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        iterator = audit.iter_events()
        next(iterator)
        if change == "gold":
            (corpus / "data/raw/gold/test.conllu").write_text(
                gold("Змінений."), encoding="utf-8"
            )
        elif change == "literature":
            (corpus / "data/raw/literature/a.txt").write_text(
                "Змінений.", encoding="utf-8"
            )
        elif change == "added":
            (corpus / "data/raw/literature/z.txt").write_text(
                "Доданий.", encoding="utf-8"
            )
        else:
            (corpus / "data/raw/literature/a.txt").unlink()
        with pytest.raises(ValueError):
            list(iterator)
        with pytest.raises(ValidationError, match="incomplete"):
            audit.report()


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/gold/bad.sqlite3",
        "models/run/index.sqlite3",
        "data/interim/a.sqlite3",
        "data/interim/run/index.db",
    ],
)
def test_invalid_index_paths(
    corpus: Path, artifact_config: PipelineConfig, path: str
) -> None:
    with (
        pytest.raises(ValueError),
        contamination_audit(artifact_config, corpus, index_path=Path(path)),
    ):
        pass


def test_existing_indexes_and_symlinks_are_preserved(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    destination = corpus / index()
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"EXISTING")
    with (
        pytest.raises(ValueError),
        contamination_audit(artifact_config, corpus, index_path=index()),
    ):
        pass
    assert destination.read_bytes() == b"EXISTING"
    destination.unlink()
    original = corpus / "data/raw/gold/test.conllu"
    before = original.read_bytes()
    destination.symlink_to(original)
    with (
        pytest.raises(ValueError),
        contamination_audit(artifact_config, corpus, index_path=index()),
    ):
        pass
    assert destination.is_symlink()
    assert original.read_bytes() == before


def test_malformed_gold_blocks_filter_before_any_eligible_yields(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/gold/test.conllu").write_text(
        word("1", "Кіт"), encoding="utf-8"
    )
    with (
        pytest.raises(ValidationError, match="missing final blank"),
        contamination_audit(artifact_config, corpus, index_path=index()),
    ):
        pytest.fail("Malformed gold cannot authorize an audit")
    assert (corpus / index()).exists()


def test_raw_inference_bypasses_pseudo_policy(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/literature/a.txt").write_text(
        "Тестовий. Тестовий.", encoding="utf-8"
    )
    assert len(list(iter_literature(artifact_config, corpus))) == 2
    result, report = run(corpus, artifact_config)
    assert all(
        isinstance(e, ContaminationExclusion) and e.reason == "gold_overlap"
        for e in result
    )
    assert report.accepted_sentence_count == 0


def test_sqlite_failure_cannot_publish_report(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        audit._connection.close()
        with pytest.raises(sqlite3.Error):
            list(audit.iter_events())
        with pytest.raises(ValidationError, match="incomplete"):
            audit.report()


def test_gold_sentences_ignore_raw_limits(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources,
            max_sentence_chars=1,
            max_sentence_tokens=1,
            max_paragraph_chars=1,
        ),
    )
    result, report = run(corpus, config)
    assert report.gold_sentence_count == 3
    assert isinstance(result[0], RawExclusion)
    assert result[0].reason == "paragraph_chars"


def test_disk_indexes_cache_and_automatic_identities(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    expected: tuple[InputIdentity, ...] = tuple(
        iter_input_identities(artifact_config, corpus)
    )
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        db = audit._connection
        assert db.execute("PRAGMA cache_size").fetchone()[0] == -2048
        assert db.execute("PRAGMA temp_store").fetchone()[0] == 1
        list(audit.iter_events())
        report = audit.report()
        assert db.execute("SELECT COUNT(*) FROM gold").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 1
    assert report.inputs == expected
    assert report.preprocessing.normalizer_version == 1
    assert report.fingerprint_version == 1
    assert report.limitations
    assert len(report.corpus_sha256) == 64


def test_excluded_source_does_not_claim_first_duplicate(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    text = "Новий текст."
    (corpus / "data/raw/literature/b.txt").write_text(text, encoding="utf-8")
    rule = KnownSourceExclusion(
        "data/raw/literature/a.txt", "val_gold", "Documented shared work"
    )
    result, report = run(corpus, artifact_config, rules=(rule,))
    assert isinstance(result[0], ContaminationExclusion)
    assert result[0].reason == "known_source"
    assert isinstance(result[1], EligibleSentence)
    assert result[1].sentence.source.source_path.endswith("/b.txt")
    assert dict(report.exclusion_counts)["literature_duplicate"] == 0


def test_policy_evidence_is_canonical_and_changes_corpus_identity(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/literature/b.txt").write_text("Інший текст.", encoding="utf-8")
    a = KnownSourceExclusion("data/raw/literature/a.txt", "val_gold", "Edition A")
    b = KnownSourceExclusion("data/raw/literature/b.txt", "test_gold", "Edition B")
    _, first = run(corpus, artifact_config, "first", (b, a))
    _, second = run(corpus, artifact_config, "second", (a, b))
    assert first == second
    _, changed = run(
        corpus, artifact_config, "changed", (replace(a, evidence="Revised evidence"), b)
    )
    assert changed.corpus_sha256 != first.corpus_sha256
    assert first.accepted_sentence_count == 0
    assert first.known_source_exclusions == (a, b)


def test_changed_corpus_rebuilds_gold_index(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    first, initial = run(corpus, artifact_config)
    assert isinstance(first[0], EligibleSentence)
    (corpus / "data/raw/gold/test.conllu").write_text(
        gold("Новий текст."), encoding="utf-8"
    )
    second, changed = run(corpus, artifact_config, "new-input")
    assert isinstance(second[0], ContaminationExclusion)
    assert second[0].reason == "gold_overlap"
    assert changed.inputs != initial.inputs
    assert changed.corpus_sha256 != initial.corpus_sha256


def test_normalized_train_mwt_patterns_are_deduplicated(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/gold/train.conllu").write_text(
        mwt("п’ять-то", "Навчальний.") + mwt("п'ять-то", "Інший навчальний."),
        encoding="utf-8",
    )
    (corpus / "data/raw/literature/a.txt").write_text("Це пʼять-то.", encoding="utf-8")
    result, report = run(corpus, artifact_config)
    assert isinstance(result[0], ContaminationExclusion)
    assert result[0].reason == "known_mwt"
    assert result[0].matched_mwt == "п’ять-то"
    assert report.mwt_pattern_count == 1


def test_gold_overlap_takes_precedence_over_mwt_and_duplicate(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/gold/train.conllu").write_text(
        mwt("аби-то", "Це аби-то."), encoding="utf-8"
    )
    (corpus / "data/raw/literature/a.txt").write_text(
        "Це аби-то. Це аби-то.", encoding="utf-8"
    )
    result, report = run(corpus, artifact_config)
    assert [e.reason for e in result if isinstance(e, ContaminationExclusion)] == [
        "gold_overlap",
        "gold_overlap",
    ]
    assert (
        dict(report.exclusion_counts)["known_mwt"]
        == dict(report.exclusion_counts)["literature_duplicate"]
        == 0
    )


def test_failure_after_eligible_yield_never_completes(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    (corpus / "data/raw/literature/b.txt").write_bytes(b"\xff")
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        stream = audit.iter_events()
        assert isinstance(next(stream), EligibleSentence)
        with pytest.raises(ValidationError, match="decoding error"):
            list(stream)
        with pytest.raises(ValidationError, match="incomplete"):
            audit.report()


def test_index_symlinked_parent_rejected(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    raw = corpus / "data/raw/gold"
    interim = corpus / "data/interim"
    interim.symlink_to(raw, target_is_directory=True)
    before = {path: path.read_bytes() for path in raw.iterdir()}
    with (
        pytest.raises(ValueError),
        contamination_audit(artifact_config, corpus, index_path=index()),
    ):
        pass
    assert {path: path.read_bytes() for path in raw.iterdir()} == before


@pytest.mark.parametrize("invalid", ["not a rule", {"bad": "rule"}, ("bad",)])
def test_malformed_rule_types(
    corpus: Path, artifact_config: PipelineConfig, invalid: object
) -> None:
    with (
        pytest.raises(ValidationError),
        contamination_audit(
            artifact_config, corpus, index_path=index(), known_source_exclusions=invalid
        ),
    ):  # type: ignore[arg-type]
        pass


def test_resource_diagnostics_and_text_volume(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources,
            max_sentence_chars=4,
            max_sentence_tokens=2,
            max_paragraph_chars=30,
        ),
    )
    text = "Тестовий.\n\nА Б!\n\nКіт.\n\nКіт."
    (corpus / "data/raw/literature/a.txt").write_text(text, encoding="utf-8")
    result, report = run(corpus, config)
    assert [
        event.reason
        for event in result
        if isinstance(event, (RawExclusion, ContaminationExclusion))
    ] == ["sentence_chars", "sentence_tokens", "literature_duplicate"]
    assert report.excluded_characters == len("Тестовий.") + len("А Б!") + len("Кіт.")
    assert report.accepted_sentence_count == 1


def test_no_report_after_successful_context_closes(
    corpus: Path, artifact_config: PipelineConfig
) -> None:
    with contamination_audit(artifact_config, corpus, index_path=index()) as audit:
        list(audit.iter_events())
        assert audit.report().accepted_sentence_count == 1
        with pytest.raises(ValidationError, match="consumed"):
            list(audit.iter_events())
    with pytest.raises(ValidationError, match="closed"):
        audit.report()
