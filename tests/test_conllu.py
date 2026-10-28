"""Offline POS-parser fixtures; no production corpus or output artifacts are used."""

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from crf.data.conllu import iter_gold_audit, iter_gold_sentences
from crf.data.identity import InputIdentity, SourceRole, digest_file
from crf.utils.config import PipelineConfig
from crf.utils.serialization import ValidationError


def row(
    identifier: str = "1", form: str = "Кіт", upos: str = "NOUN", misc: str = "_"
) -> str:
    return f"{identifier}\t{form}\t_\t{upos}\t_\t_\t0\troot\t_\t{misc}\n"


def source(
    root: Path,
    content: str | bytes,
    role: SourceRole = "train_gold",
) -> InputIdentity:
    relative = f"data/raw/gold/{role.removesuffix('_gold')}.conllu"
    path = root / relative
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    digest = digest_file(path)
    return InputIdentity(1, role, relative, digest.sha256, digest.size_bytes, "utf-8")


def test_words_provenance_and_original_metadata(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    content = (
        "# sent_id = uk-1\n# text = Кіт!\n# custom = preserved\n"
        + row(misc="SpaceAfter=No")
        + row("2", "!", "PUNCT")
        + "\n\n"
        + row(form="_", upos="X")
        + "\n"
    )
    identity = source(artifact_repository, content)
    sentences = list(
        iter_gold_sentences(identity, artifact_config, artifact_repository)
    )
    first, second = sentences
    assert first.original_upos == first.target_labels == ("NOUN", "PUNCT")
    assert first.audit.original_forms == ("Кіт", "!")
    assert first.audit.token_ids == (1, 2)
    assert first.audit.source == identity
    assert first.audit.sent_id == "uk-1"
    assert first.audit.surface_text == "Кіт!"
    assert first.audit.surface_text_origin == "original"
    assert first.comments == (
        "# sent_id = uk-1",
        "# text = Кіт!",
        "# custom = preserved",
    )
    assert second.audit.original_forms == ("_",)
    assert second.audit.surface_text_origin == "reconstructed"
    assert [item.audit.sentence_ordinal for item in sentences] == [1, 2]
    assert second.audit.line_number == 8
    expected = hashlib.sha256(
        json.dumps(
            ["train_gold", identity.source_path, identity.source_sha256, 1],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert first.audit.sentence_id == expected
    assert (artifact_repository / identity.source_path).read_bytes() == content.encode()
    with pytest.raises(FrozenInstanceError):
        first.original_upos = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.audit.surface_text = "changed"  # type: ignore[misc]


def test_multiword_reconstruction_and_empty_nodes(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    content = (
        row("1-2", "аби-то", "_", "SpaceAfter=No")
        + row("1", "аби", "SCONJ")
        + row("1.1", "ghost", "_")
        + row("1.10", "ghost2", "_")
        + row("2", "то", "PART")
        + row("3", ",", "PUNCT")
        + row("4", "кіт")
        + "\n"
    )
    identity = source(artifact_repository, content)
    parsed = next(iter_gold_sentences(identity, artifact_config, artifact_repository))
    audit = parsed.audit
    assert audit.token_ids == (1, 2, 3, 4)
    assert audit.original_forms == ("аби", "то", ",", "кіт")
    assert parsed.original_upos == ("SCONJ", "PART", "PUNCT", "NOUN")
    assert audit.surface_text == "аби-то, кіт"
    assert audit.surface_rows[0].token_id == (1, "-", 2)
    assert audit.surface_rows[3].token_id == (1, ".", 10)
    assert list(iter_gold_audit(identity, artifact_config, artifact_repository)) == [
        audit
    ]


def test_document_and_paragraph_continuity(
    artifact_repository: Path, artifact_config: PipelineConfig
) -> None:
    content = (
        "# newdoc id = d1\n# newpar id = p1\n"
        + row()
        + "\n"
        + row()
        + "\n# newpar id = p2\n"
        + row()
        + "\n"
        + "# newdoc id = d2\n"
        + row()
        + "\n# newdoc\n"
        + row()
        + "\n"
    )
    identity = source(artifact_repository, content)
    audits = list(iter_gold_audit(identity, artifact_config, artifact_repository))
    assert [(x.document_id, x.paragraph_id) for x in audits] == [
        ("d1", "p1"),
        ("d1", "p1"),
        ("d1", "p2"),
        ("d2", None),
        (None, None),
    ]


@pytest.mark.parametrize(
    ("misc", "next_misc", "expected"),
    [
        ("_", "_", "Кіт спить"),
        ("SpaceAfter=No", "_", "Кітспить"),
        (r"SpacesAfter=\s\t\n", "_", "Кіт \t\nспить"),
        ("_", r"SpacesBefore=\s\s", "Кіт  спить"),
        ("_", r"SpacesAfter=\n", "Кіт спить\n"),
        ("Note=secret|SpaceAfter=No", "_", "Кітспить"),
    ],
)
def test_spacing(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    misc: str,
    next_misc: str,
    expected: str,
) -> None:
    identity = source(
        artifact_repository,
        row(misc=misc) + row("2", "спить", "VERB", next_misc) + "\n",
    )
    audit = next(iter_gold_audit(identity, artifact_config, artifact_repository))
    assert audit.surface_text == expected
    assert "secret" not in repr(audit)


def test_audit_leaves_all_annotation_columns_uninterpreted(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    content = (
        "# annotation = SECRET_COMMENT\n# text = Кіт\n"
        "1\tКіт\tSECRET_LEMMA\tINVALID_UPOS\tSECRET_XPOS\tBAD_FEATS\tBAD_HEAD"
        "\tBAD_DEPREL\tBAD_DEPS\tSecret=HIDDEN\n\n"
    )
    identity = source(artifact_repository, content, "test_gold")
    audit = next(iter_gold_audit(identity, artifact_config, artifact_repository))
    assert audit.surface_text == "Кіт"
    assert set(asdict(audit)) == {
        "sentence_id",
        "source",
        "sentence_ordinal",
        "line_number",
        "token_ids",
        "original_forms",
        "surface_text",
        "surface_text_origin",
        "sent_id",
        "document_id",
        "paragraph_id",
        "surface_rows",
    }
    assert not any(
        value in repr(audit) for value in ("SECRET", "INVALID_UPOS", "BAD_", "HIDDEN")
    )
    with pytest.raises(ValidationError, match="original UPOS"):
        list(
            iter_gold_sentences(
                identity, artifact_config, artifact_repository, final_evaluation=True
            )
        )


@pytest.mark.parametrize("flag", [False, 1, "yes", None])
def test_test_label_gate_precedes_any_source_read(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    flag: object,
) -> None:
    identity = InputIdentity(1, "test_gold", "missing", "0" * 64, 0, "utf-8")
    with pytest.raises(ValidationError, match="final evaluation|expected boolean"):
        list(
            iter_gold_sentences(
                identity,
                artifact_config,
                artifact_repository,
                final_evaluation=flag,  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("role", ["train_gold", "val_gold", "test_gold"])
def test_all_splits_and_explicit_final_opt_in(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    role: SourceRole,
) -> None:
    identity = source(artifact_repository, row() + "\n", role)
    labeled = list(
        iter_gold_sentences(
            identity,
            artifact_config,
            artifact_repository,
            final_evaluation=role == "test_gold",
        )
    )
    assert labeled[0].audit.source.source_role == role
    assert labeled[0].target_labels == ("NOUN",)


@pytest.mark.parametrize(
    ("content", "line", "message"),
    [
        (row(), 1, "final blank"),
        (row() + " ", 2, "unterminated blank"),
        ("# text = Кіт\n\n", 1, "no integer"),
        (row("1-2", "MWT", "_") + "\n", 1, "no integer"),
        (row("0.1", "empty", "_") + "\n", 1, "no integer"),
        (row() + "# text = late\n\n", 2, "comments must precede"),
        (row().replace("\t", "  ") + "\n", 1, "ten nonempty"),
        (row().replace("\t_\tNOUN", "\t\tNOUN") + "\n", 1, "ten nonempty"),
        (row().rstrip("\n") + "\textra\n\n", 1, "ten nonempty"),
        (row("2") + "\n", 1, "consecutive"),
        (row() + row("3") + "\n", 2, "consecutive"),
        (row() + row() + "\n", 2, "duplicate"),
        (row("0") + "\n", 1, "positive integer"),
        (row("01") + "\n", 1, "invalid CoNLL-U ID"),
        (row("-1") + "\n", 1, "invalid CoNLL-U ID"),
        (row("1.0") + "\n", 1, "invalid CoNLL-U ID"),
        (row("1.01") + "\n", 1, "invalid CoNLL-U ID"),
        (row("2-1") + "\n", 1, "invalid CoNLL-U ID"),
        (row("1-1") + "\n", 1, "range/empty-node ID"),
        (row("1-3") + row() + row("2") + "\n", 3, "exceeds sentence"),
        (row() + row("1-2") + row("2") + "\n", 2, "misplaced"),
        (row("1-2") + row() + row("2-3") + "\n", 3, "overlapping"),
        (row("1.1") + row() + "\n", 1, "word anchor"),
        (row() + row("1.1") + row("1.1") + "\n", 3, "duplicate"),
        ("# sent_id = a\n# sent_id = b\n" + row() + "\n", 2, "duplicate"),
        ("# text = \n" + row() + "\n", 1, "empty text"),
        (row(misc="SpaceAfter=Yes") + "\n", 1, "SpaceAfter must"),
        (row(misc="SpaceAfter=No|SpaceAfter=No") + "\n", 1, "duplicate spacing"),
        (row(misc=r"SpacesAfter=\q") + "\n", 1, "spacing escape"),
        (b"\xef\xbb\xbf" + (row() + "\n").encode(), 1, "BOM"),
        (row().encode() + b"2\t\xff\n\n", 2, "UTF-8"),
        (row(form="bad\rform") + "\n", 1, "unexpected CR"),
    ],
)
def test_framing_and_surface_failures_apply_to_both_interfaces(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    content: str | bytes,
    line: int,
    message: str,
) -> None:
    identity = source(artifact_repository, content)
    for reader in (iter_gold_sentences, iter_gold_audit):
        with pytest.raises(ValidationError, match=message) as exc:
            list(reader(identity, artifact_config, artifact_repository))
        assert f"{identity.source_path}:{line}:" in str(exc.value)


@pytest.mark.parametrize("upos", ["_", "FUNC", "noun", "UNKNOWN"])
def test_invalid_source_labels_fail_before_mapping(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    upos: str,
) -> None:
    identity = source(artifact_repository, row(upos=upos) + "\n")
    with pytest.raises(ValidationError, match="original UPOS"):
        list(iter_gold_sentences(identity, artifact_config, artifact_repository))
    assert (
        len(list(iter_gold_audit(identity, artifact_config, artifact_repository))) == 1
    )


def test_streams_complete_sentences_before_later_errors(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    identity = source(artifact_repository, row() + "\n" + row("2") + "\n")
    stream = iter_gold_sentences(identity, artifact_config, artifact_repository)
    assert next(stream).audit.sentence_ordinal == 1
    with pytest.raises(ValidationError, match="consecutive"):
        next(stream)


def test_existing_validated_mapping_preserves_original_labels(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    config = replace(
        artifact_config,
        tag_mapping=tuple(
            (label, "FUNC") for label in ("ADP", "CCONJ", "SCONJ", "PART")
        ),
    )
    identity = source(artifact_repository, row(upos="ADP") + row("2") + "\n")
    parsed = next(iter_gold_sentences(identity, config, artifact_repository))
    assert parsed.original_upos == ("ADP", "NOUN")
    assert parsed.target_labels == ("FUNC", "NOUN")
    bad = source(artifact_repository, row(upos="FUNC") + "\n")
    with pytest.raises(ValidationError, match="original UPOS"):
        list(iter_gold_sentences(bad, config, artifact_repository))


def test_gold_is_not_subject_to_raw_sentence_limits(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    config = replace(
        artifact_config,
        resources=replace(
            artifact_config.resources, max_sentence_tokens=1, max_sentence_chars=1
        ),
    )
    identity = source(
        artifact_repository,
        "".join(row(str(i), "ґ" * 100) for i in range(1, 301)) + "\n",
    )
    sentence = next(iter_gold_sentences(identity, config, artifact_repository))
    assert len(sentence.audit.original_forms) == 300
    assert len(sentence.audit.surface_text) == 30299


def test_crlf_unicode_and_empty_source(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    identity = source(
        artifact_repository,
        ("\n" + row(form="Ґїієґ’\ufeff\u200b\u0301") + "\n").replace("\n", "\r\n"),
    )
    sentence = next(iter_gold_sentences(identity, artifact_config, artifact_repository))
    assert sentence.audit.original_forms == ("Ґїієґ’\ufeff\u200b\u0301",)
    assert sentence.audit.line_number == 2
    empty = source(artifact_repository, "\n\n")
    assert list(iter_gold_sentences(empty, artifact_config, artifact_repository)) == []


def test_identity_mismatch_missing_and_wrong_role(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    identity = source(artifact_repository, row() + "\n")
    for reader in (iter_gold_sentences, iter_gold_audit):
        with pytest.raises(ValidationError, match="checksum/size mismatch"):
            list(
                reader(
                    replace(identity, source_sha256="0" * 64),
                    artifact_config,
                    artifact_repository,
                )
            )
        with pytest.raises(ValidationError, match="gold source role"):
            list(
                reader(
                    replace(identity, source_role="literature"),
                    artifact_config,
                    artifact_repository,
                )
            )
    (artifact_repository / identity.source_path).unlink()
    with pytest.raises(ValidationError, match="Invalid source location"):
        list(iter_gold_audit(identity, artifact_config, artifact_repository))


def test_source_change_during_iteration_fails_at_eof(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
) -> None:
    identity = source(artifact_repository, row() + "\n")
    stream = iter_gold_audit(identity, artifact_config, artifact_repository)
    next(stream)
    with (artifact_repository / identity.source_path).open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValidationError, match="source changed while parsing"):
        list(stream)
