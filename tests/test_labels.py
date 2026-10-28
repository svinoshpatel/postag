"""Specification-derived label-space checks using only deterministic offline fixtures."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from crf.data.conllu import iter_gold_audit, iter_gold_sentences
from crf.data.identity import (
    InputIdentity,
    SourceRole,
    digest_file,
    sampling_priority,
    sentence_id,
)
from crf.data.labels import (
    SERVICE_TAGS,
    UPOS,
    TagMapping,
    TagMappingError,
    validate_upos,
)
from crf.data.records import parse_sentence_record
from crf.features.surface import sentence_features
from crf.utils.config import SERVICE_TAGS as CONFIG_SERVICE_TAGS
from crf.utils.config import UPOS as CONFIG_UPOS
from crf.utils.config import ConfigError, PipelineConfig, parse_config
from crf.utils.serialization import (
    ValidationError,
    canonical_config_bytes,
    config_sha256,
    jsonl_bytes,
    loads_json,
)

UD_TAGS = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)
CONSOLIDATED_TAGS = (
    "ADJ",
    "ADV",
    "AUX",
    "DET",
    "FUNC",
    "INTJ",
    "NOUN",
    "NUM",
    "PRON",
    "PROPN",
    "PUNCT",
    "SYM",
    "VERB",
    "X",
)
MAPPED_SEQUENCE = (
    "ADJ",
    "FUNC",
    "ADV",
    "AUX",
    "FUNC",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "FUNC",
    "PRON",
    "PROPN",
    "PUNCT",
    "FUNC",
    "SYM",
    "VERB",
    "X",
)
SERVICES = ("ADP", "CCONJ", "SCONJ", "PART")


def mapping(target: str = "FUNC") -> dict[str, str]:
    return {label: target for label in SERVICES}


def record_value(source: InputIdentity, *, pseudo: bool = False) -> dict[str, object]:
    """Labels are overridden by each test; forms need no linguistic normalization."""
    return {
        "schema_version": 1,
        "record_type": "pseudo" if pseudo else "gold",
        "sentence_id": sentence_id(
            source.source_role, source.source_path, source.source_sha256, 1
        ),
        "source_role": source.source_role,
        "source_path": source.source_path,
        "source_sha256": source.source_sha256,
        "sentence_ordinal": 1,
        "paragraph_ordinal": 1 if pseudo else None,
        "surface_text": "раз два",
        "surface_text_origin": "original",
        "token_ids": [1, 2],
        "original_forms": ["раз", "два"],
        "normalized_forms": ["раз", "два"],
        "token_spans": [None, None],
        "original_upos": None if pseudo else ["ADP", "NOUN"],
        "target_labels": ["FUNC", "NOUN"],
        "pseudo": {
            "token_confidences": [0.95, 0.98],
            "min_confidence": 0.95,
            "sentence_fingerprint": "a" * 64,
            "sampling_priority": sampling_priority(42, "a" * 64),
            "gate_model_sha256": "b" * 64,
        }
        if pseudo
        else None,
    }


def test_fixed_inventories_and_compatible_constants() -> None:
    disabled = TagMapping()
    enabled = TagMapping.from_dict(mapping())
    assert disabled.target_inventory == UD_TAGS
    assert enabled.target_inventory == CONSOLIDATED_TAGS
    assert len(disabled.target_inventory) == 17
    assert len(enabled.target_inventory) == 14
    assert disabled.map_upos(UD_TAGS) == UD_TAGS
    assert enabled.map_upos(UD_TAGS) == MAPPED_SEQUENCE
    assert enabled.validate_targets(MAPPED_SEQUENCE) == MAPPED_SEQUENCE
    assert CONFIG_UPOS is UPOS
    assert CONFIG_SERVICE_TAGS is SERVICE_TAGS


def test_immutable_snapshots_canonical_order_and_repetitions() -> None:
    supplied = mapping()
    labels = ["PART", "PART", "X", "ADP"]
    enabled = TagMapping.from_dict(dict(reversed(tuple(supplied.items()))))
    assert enabled.entries == tuple(sorted(supplied.items()))
    assert enabled.map_upos(labels) == ("FUNC", "FUNC", "X", "FUNC")
    assert labels == ["PART", "PART", "X", "ADP"]
    supplied["PART"] = "OTHER"
    assert enabled.target_inventory == CONSOLIDATED_TAGS
    with pytest.raises(FrozenInstanceError):
        enabled.entries = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "FUNC",
        1,
        True,
        {1: "FUNC"},
        {"ADP": "FUNC"},
        mapping() | {"NOUN": "FUNC"},
        mapping() | {"PART": "OTHER"},
    ],
)
def test_reject_invalid_mapping_objects(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    bad: object,
) -> None:
    with pytest.raises(TagMappingError, match="tag_mapping"):
        TagMapping.from_dict(bad)
    with pytest.raises(ConfigError, match="tag_mapping"):
        parse_config(
            artifact_config.to_dict() | {"tag_mapping": bad},
            repository_root=artifact_repository,
        )


@pytest.mark.parametrize(
    "target",
    [
        "",
        " ",
        "two words",
        "FUNC\t",
        "\nFUNC",
        "FUNC\u00a0",
        "FUNC\u202f",
        "FUNC\x00",
        "\ud800",
        None,
        False,
        1,
        [],
        {},
        *UD_TAGS,
    ],
)
def test_reject_invalid_target_names(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    target: object,
) -> None:
    raw_mapping = {tag: target for tag in SERVICES}
    with pytest.raises(TagMappingError, match="tag_mapping"):
        TagMapping.from_dict(raw_mapping)
    with pytest.raises(ConfigError, match="tag_mapping"):
        parse_config(
            artifact_config.to_dict() | {"tag_mapping": raw_mapping},
            repository_root=artifact_repository,
        )


@pytest.mark.parametrize(
    "entries",
    [
        [],
        (("ADP", "FUNC"),),
        (("ADP", "FUNC"),) * 4,
        tuple(mapping().items()) + (("ADP", "FUNC"),),
        (("ADP", "FUNC", "extra"),),
        (None,),
        (["ADP", "FUNC"],),
    ],
)
def test_reject_malformed_direct_tuple_inputs(entries: object) -> None:
    with pytest.raises(TagMappingError, match="tag_mapping"):
        TagMapping(entries)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["FUNC", "_", "noun", "", None, True, [], "NOUN "])
def test_original_labels_are_validated_before_mapping(bad: object) -> None:
    with pytest.raises(TagMappingError, match="original UPOS"):
        validate_upos(bad)
    with pytest.raises(TagMappingError, match="original UPOS"):
        TagMapping.from_dict(mapping()).map_upos(["ADP", bad])  # type: ignore[list-item]


@pytest.mark.parametrize("labels", [[], (), "NOUN", b"NOUN", None, 1, iter(["NOUN"])])
def test_sequence_boundaries(labels: object) -> None:
    enabled = TagMapping.from_dict(mapping())
    for operation in (enabled.map_upos, enabled.validate_targets):
        with pytest.raises(TagMappingError, match="nonempty sequence"):
            operation(labels)  # type: ignore[arg-type]


@pytest.mark.parametrize("label", [*SERVICES, "OTHER", "func", None, True, []])
def test_targets_are_checked_without_remapping(label: object) -> None:
    with pytest.raises(TagMappingError, match="target inventory"):
        TagMapping.from_dict(mapping()).validate_targets([label])  # type: ignore[list-item]
    with pytest.raises(TagMappingError, match="target inventory"):
        TagMapping().validate_targets(["FUNC"])


@pytest.mark.parametrize("target", ["FUNC", "службове", "func", "Е\u0301"])
def test_custom_targets_and_config_serialization(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    target: str,
) -> None:
    raw = artifact_config.to_dict() | {"tag_mapping": mapping(target)}
    config = parse_config(raw, repository_root=artifact_repository)
    assert config.label_space.map_upos(["ADP"]) == (target,)
    assert config.target_inventory == tuple(
        sorted(
            (
                "ADJ",
                "ADV",
                "AUX",
                "DET",
                target,
                "INTJ",
                "NOUN",
                "NUM",
                "PRON",
                "PROPN",
                "PUNCT",
                "SYM",
                "VERB",
                "X",
            )
        )
    )
    assert "target_inventory" not in config.to_dict()
    restored = parse_config(
        loads_json(canonical_config_bytes(config)), repository_root=artifact_repository
    )
    assert restored == config
    assert restored.target_inventory == config.target_inventory
    assert config_sha256(config) != config_sha256(artifact_config)
    reordered = replace(config, tag_mapping=tuple(reversed(config.tag_mapping)))
    assert canonical_config_bytes(reordered) == canonical_config_bytes(config)


@pytest.mark.parametrize("target", [None, "FUNC", "СЛУЖБОВЕ"])
@pytest.mark.parametrize("role", ["train_gold", "val_gold", "test_gold"])
def test_loader_and_record_round_trip_use_same_targets_for_every_split(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    role: SourceRole,
    target: str | None,
) -> None:
    config = parse_config(
        artifact_config.to_dict()
        | {"tag_mapping": {} if target is None else mapping(target)},
        repository_root=artifact_repository,
    )
    path = artifact_repository / getattr(config.data, role)
    content = (
        "# sent_id = fixed\n"
        + "".join(
            f"{index}\tслово{index}\t_\t{label}\t_\t_\t0\troot\t_\t_\n"
            for index, label in enumerate(UD_TAGS, 1)
        )
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    digest = digest_file(path)
    source = InputIdentity(
        1,
        role,
        getattr(config.data, role).as_posix(),
        digest.sha256,
        digest.size_bytes,
        "utf-8",
    )
    if role == "test_gold":
        with pytest.raises(ValidationError, match="final evaluation"):
            list(iter_gold_sentences(source, config, artifact_repository))
    (parsed,) = iter_gold_sentences(
        source, config, artifact_repository, final_evaluation=role == "test_gold"
    )
    expected = (
        UD_TAGS
        if target is None
        else tuple(target if label == "FUNC" else label for label in MAPPED_SEQUENCE)
    )
    assert parsed.original_upos == UD_TAGS
    assert parsed.target_labels == expected
    assert config.label_space.validate_targets(parsed.target_labels) == expected
    audit = parsed.audit
    value = record_value(source) | {
        "surface_text": audit.surface_text,
        "surface_text_origin": audit.surface_text_origin,
        "token_ids": list(audit.token_ids),
        "original_forms": list(audit.original_forms),
        "normalized_forms": list(audit.original_forms),
        "token_spans": [None] * len(UD_TAGS),
        "original_upos": list(parsed.original_upos),
        "target_labels": list(parsed.target_labels),
    }
    restored = parse_sentence_record(
        loads_json(jsonl_bytes(value)),
        config=config,
        source=source,
        final_evaluation=role == "test_gold",
    )
    assert restored.to_dict() == value
    assert restored.original_upos == UD_TAGS
    assert restored.target_labels == expected
    assert sentence_features(restored.feature_forms) == sentence_features(
        audit.original_forms
    )
    assert list(iter_gold_audit(source, config, artifact_repository)) == [audit]
    assert path.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize("pseudo", [False, True])
def test_mapping_changes_reject_old_records_without_changing_confidences(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    pseudo: bool,
) -> None:
    source = artifact_sources[-1 if pseudo else 0]
    value = record_value(source, pseudo=pseudo)
    config = parse_config(
        artifact_config.to_dict() | {"tag_mapping": mapping()},
        repository_root=artifact_repository,
    )
    record = parse_sentence_record(value, config=config, source=source)
    assert record.target_labels == ("FUNC", "NOUN")
    if pseudo:
        assert record.pseudo is not None
        assert record.pseudo.token_confidences == (0.95, 0.98)
    else:
        assert record.original_upos == ("ADP", "NOUN")
    for changed in (
        artifact_config,
        replace(config, tag_mapping=tuple(mapping("OTHER").items())),
    ):
        with pytest.raises(ValidationError, match="target_labels"):
            parse_sentence_record(record.to_dict(), config=changed, source=source)
    with pytest.raises(ValidationError, match="target_labels"):
        parse_sentence_record(
            value | {"target_labels": ["ADP", "NOUN"]}, config=config, source=source
        )
    if not pseudo:
        with pytest.raises(ValidationError, match="original UPOS"):
            parse_sentence_record(
                value
                | {
                    "original_upos": ["FUNC", "NOUN"],
                    "target_labels": ["INVALID", "NOUN"],
                },
                config=config,
                source=source,
            )


def test_invalid_direct_config_fails_consumers_but_does_not_affect_audit(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
) -> None:
    config = replace(artifact_config, tag_mapping=(("ADP", "FUNC"),))
    with pytest.raises(ConfigError, match="tag_mapping"):
        config.to_dict()
    with pytest.raises(ConfigError, match="tag_mapping"):
        _ = config.target_inventory
    with pytest.raises(ValidationError, match="tag_mapping"):
        parse_sentence_record(
            record_value(artifact_sources[0]), config=config, source=artifact_sources[0]
        )
    path = artifact_repository / config.data.test_gold
    content = "1\tслово\topaque\tPOISON\topaque\topaque\topaque\topaque\topaque\t_\n\n"
    path.write_text(content, encoding="utf-8")
    digest = digest_file(path)
    source = InputIdentity(
        1,
        "test_gold",
        config.data.test_gold.as_posix(),
        digest.sha256,
        digest.size_bytes,
        "utf-8",
    )
    (audit,) = iter_gold_audit(source, config, artifact_repository)
    assert audit.original_forms == ("слово",)
    assert "POISON" not in repr(audit)
    with pytest.raises(ValidationError, match="tag_mapping"):
        list(
            iter_gold_sentences(
                source, config, artifact_repository, final_evaluation=True
            )
        )
