"""Schema-derived gold/pseudo fixtures; no production annotations are loaded."""

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from crf.data.identity import InputIdentity, sampling_priority, sentence_id
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.features.surface import sentence_features
from crf.utils.config import PipelineConfig
from crf.utils.serialization import ValidationError, iter_jsonl, jsonl_bytes, loads_json


def sentence(source: InputIdentity, *, pseudo: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "pseudo" if pseudo else "gold",
        "sentence_id": sentence_id(
            source.source_role, source.source_path, source.source_sha256, 7
        ),
        "source_path": source.source_path,
        "source_sha256": source.source_sha256,
        "source_role": source.source_role,
        "sentence_ordinal": 7,
        "paragraph_ordinal": 2 if pseudo else None,
        "surface_text": "Кіт _",
        "surface_text_origin": "original",
        "token_ids": [1, 2],
        "original_forms": ["Кіт", "_"],
        "normalized_forms": ["Кіт", "_"],
        "token_spans": [[0, 3], [4, 5]] if pseudo else [None, None],
        "original_upos": None if pseudo else ["NOUN", "X"],
        "target_labels": ["NOUN", "X"],
        "pseudo": {
            "token_confidences": [0.9000000000000001, 0.98],
            "min_confidence": 0.9000000000000001,
            "sentence_fingerprint": "a" * 64,
            "sampling_priority": sampling_priority(42, "a" * 64),
            "gate_model_sha256": "b" * 64,
        }
        if pseudo
        else None,
    }


@pytest.mark.parametrize("pseudo", [False, True])
def test_round_trip_immutable_and_feature_boundary(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    pseudo: bool,
) -> None:
    source = artifact_sources[-1 if pseudo else 0]
    value = sentence(source, pseudo=pseudo)
    record = parse_sentence_record(
        loads_json(jsonl_bytes(value)),
        config=artifact_config,
        source=source,
        decoded_source_length=5,
    )
    assert record.to_dict() == value
    assert isinstance(record, SentenceRecord)
    assert record.sentence_ordinal == 7
    assert record.original_forms[-1] == "_"
    assert record.feature_forms == ("Кіт", "_")
    features = sentence_features(record.feature_forms)
    assert features[0]["word.lower"] == "кіт"
    assert not any(
        key in feature
        for feature in features
        for key in ("source_path", "original_upos", "target_labels")
    )
    with pytest.raises(FrozenInstanceError):
        record.sentence_ordinal = 1
    value["original_forms"][0] = "changed"
    assert record.original_forms[0] == "Кіт"
    if pseudo:
        assert record.pseudo.min_confidence == 0.9000000000000001


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("record_type", "other"),
        ("sentence_id", "A" * 64),
        ("sentence_id", "0" * 64),
        ("source_path", "other"),
        ("source_sha256", "0" * 64),
        ("source_role", "val_gold"),
        ("sentence_ordinal", True),
        ("sentence_ordinal", 0),
        ("sentence_ordinal", 8),
        ("surface_text", ""),
        ("surface_text", 1),
        ("surface_text_origin", "invented"),
        ("token_ids", []),
        ("token_ids", [1, 3]),
        ("token_ids", [True, 2]),
        ("token_ids", [1.0, 2]),
        ("token_ids", ["1", "2"]),
        ("token_ids", [0, 1]),
        ("original_forms", ["Кіт"]),
        ("original_forms", ["Кіт", ""]),
        ("normalized_forms", ["Кіт", None]),
        ("normalized_forms", "Кіт _"),
        ("token_spans", [None]),
        ("token_spans", [[0, 3, 4], None]),
        ("token_spans", [[True, 3], None]),
        ("token_spans", [[0, 6], None]),
        ("token_spans", [[-1, 3], None]),
        ("token_spans", [[2, 2], None]),
        ("token_spans", [[0, 3], [2, 4]]),
        ("token_spans", [[3, 4], [0, 3]]),
        ("original_upos", ["NOUN", "UNKNOWN"]),
        ("original_upos", ["NOUN"]),
        ("original_upos", None),
        ("target_labels", ["NOUN", "UNKNOWN"]),
        ("target_labels", ["NOUN", "VERB"]),
        ("target_labels", []),
        ("paragraph_ordinal", 1),
        ("pseudo", {}),
    ],
)
def test_gold_failures(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    field: str,
    bad: object,
) -> None:
    source = artifact_sources[0]
    with pytest.raises(ValidationError):
        parse_sentence_record(
            sentence(source) | {field: bad},
            config=artifact_config,
            source=source,
            decoded_source_length=5,
        )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("token_confidences", []),
        ("token_confidences", [True, 0.9]),
        ("token_confidences", [float("nan"), 0.9]),
        ("token_confidences", [1.01, 0.9]),
        ("token_confidences", [-0.1, 0.9]),
        ("token_confidences", ["0.9", 0.9]),
        ("min_confidence", 0.9),
        ("min_confidence", True),
        ("sampling_priority", "0" * 64),
        ("sentence_fingerprint", "a" * 63),
        ("gate_model_sha256", "Z" * 64),
    ],
)
def test_pseudo_metadata_failures(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    field: str,
    bad: object,
) -> None:
    source = artifact_sources[-1]
    value = sentence(source, pseudo=True)
    value["pseudo"][field] = bad
    with pytest.raises(ValidationError):
        parse_sentence_record(
            value, config=artifact_config, source=source, decoded_source_length=5
        )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("original_upos", ["NOUN", "X"]),
        ("surface_text_origin", "reconstructed"),
        ("paragraph_ordinal", None),
        ("paragraph_ordinal", True),
        ("pseudo", None),
        ("record_type", "gold"),
    ],
)
def test_pseudo_nullability(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    field: str,
    bad: object,
) -> None:
    source = artifact_sources[-1]
    with pytest.raises(ValidationError):
        parse_sentence_record(
            sentence(source, pseudo=True) | {field: bad},
            config=artifact_config,
            source=source,
            decoded_source_length=5,
        )


def test_exact_sentence_and_pseudo_fields(
    artifact_config: PipelineConfig, artifact_sources: tuple[InputIdentity, ...]
) -> None:
    source = artifact_sources[-1]
    value = sentence(source, pseudo=True)
    for key in value:
        invalid = {name: item for name, item in value.items() if name != key}
        with pytest.raises(ValidationError):
            parse_sentence_record(
                invalid, config=artifact_config, source=source, decoded_source_length=5
            )
    for key in value["pseudo"]:
        invalid = copy.deepcopy(value)
        del invalid["pseudo"][key]
        with pytest.raises(ValidationError):
            parse_sentence_record(
                invalid, config=artifact_config, source=source, decoded_source_length=5
            )
    for target in (value, value["pseudo"]):
        target["extra"] = None
        with pytest.raises(ValidationError):
            parse_sentence_record(
                value, config=artifact_config, source=source, decoded_source_length=5
            )
        del target["extra"]


def test_mapping_and_original_inventory(
    artifact_config: PipelineConfig, artifact_sources: tuple[InputIdentity, ...]
) -> None:
    config = replace(
        artifact_config,
        tag_mapping=tuple((tag, "FUNC") for tag in ("ADP", "CCONJ", "SCONJ", "PART")),
    )
    source = artifact_sources[0]
    value = sentence(source) | {
        "original_upos": ["ADP", "PART"],
        "target_labels": ["FUNC", "FUNC"],
    }
    record = parse_sentence_record(value, config=config, source=source)
    assert record.original_upos == ("ADP", "PART")
    assert record.target_labels == ("FUNC", "FUNC")
    for invalid in (
        value | {"original_upos": ["FUNC", "PART"]},
        value | {"target_labels": ["ADP", "PART"]},
    ):
        with pytest.raises(ValidationError):
            parse_sentence_record(invalid, config=config, source=source)


def test_test_labels_restricted(
    artifact_config: PipelineConfig, artifact_sources: tuple[InputIdentity, ...]
) -> None:
    source = artifact_sources[2]
    value = sentence(source)
    with pytest.raises(ValidationError, match="final evaluation"):
        parse_sentence_record(value, config=artifact_config, source=source)
    assert (
        parse_sentence_record(
            value, config=artifact_config, source=source, final_evaluation=True
        ).source_role
        == "test_gold"
    )


def test_span_context_and_diagnostics(
    tmp_path: Path,
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
) -> None:
    source = artifact_sources[-1]
    value = sentence(source, pseudo=True)
    for length in (None, True, -1, 4):
        with pytest.raises(ValidationError):
            parse_sentence_record(
                value,
                config=artifact_config,
                source=source,
                decoded_source_length=length,
            )
    value["token_ids"] = [1, 2, 3]
    for field in ("original_forms", "normalized_forms", "target_labels"):
        value[field].append(value[field][-1])
    value["token_spans"] = [[0, 3], None, [2, 5]]
    value["pseudo"]["token_confidences"].append(0.98)
    with pytest.raises(ValidationError, match="overlapping"):
        parse_sentence_record(
            value, config=artifact_config, source=source, decoded_source_length=5
        )
    path = tmp_path / "sentences.jsonl"
    path.write_bytes(jsonl_bytes(sentence(source, pseudo=True)) + jsonl_bytes(value))

    def validate(raw: object) -> SentenceRecord:
        return parse_sentence_record(
            raw, config=artifact_config, source=source, decoded_source_length=5
        )

    records = iter_jsonl(path, validate)
    assert next(records).pseudo.min_confidence == 0.9000000000000001
    with pytest.raises(ValidationError, match=r"sentences.jsonl: record/line 2"):
        next(records)


def test_probability_endpoints_global_spans_and_reconstruction(
    artifact_config: PipelineConfig, artifact_sources: tuple[InputIdentity, ...]
) -> None:
    source = artifact_sources[-1]
    value = sentence(source, pseudo=True)
    value["pseudo"]["token_confidences"] = [0, 1]
    value["pseudo"]["min_confidence"] = 0
    value["token_spans"] = [[100, 103], [104, 105]]
    record = parse_sentence_record(
        value, config=artifact_config, source=source, decoded_source_length=105
    )
    assert record.pseudo.token_confidences == (0.0, 1.0)
    assert record.token_spans == ((100, 103), (104, 105))
    gold = artifact_sources[0]
    assert (
        parse_sentence_record(
            sentence(gold) | {"surface_text_origin": "reconstructed"},
            config=artifact_config,
            source=gold,
        ).surface_text_origin
        == "reconstructed"
    )


def test_pseudo_target_inventory_and_seed_binding(
    artifact_config: PipelineConfig, artifact_sources: tuple[InputIdentity, ...]
) -> None:
    source = artifact_sources[-1]
    value = sentence(source, pseudo=True)
    config = replace(
        artifact_config,
        tag_mapping=tuple((tag, "FUNC") for tag in ("ADP", "CCONJ", "SCONJ", "PART")),
    )
    value["target_labels"] = ["FUNC", "X"]
    assert parse_sentence_record(
        value, config=config, source=source, decoded_source_length=5
    ).target_labels == ("FUNC", "X")
    with pytest.raises(ValidationError, match="target inventory"):
        parse_sentence_record(
            value | {"target_labels": ["ADP", "X"]},
            config=config,
            source=source,
            decoded_source_length=5,
        )
    with pytest.raises(ValidationError, match="seed/fingerprint"):
        parse_sentence_record(
            value,
            config=replace(config, seed=43),
            source=source,
            decoded_source_length=5,
        )


@pytest.mark.parametrize("pseudo", [False, True])
@pytest.mark.parametrize(
    ("original", "normalized", "bad"),
    [
        ("сім’я", "сім'я", "сім’я"),
        ("І\u0308", "Ї", "І\u0308"),
        ("ҐАНОК", "ҐАНОК", "ґанок"),
        ("а\u00a0б", "а б", "а\u00a0б"),
        ("кіт\u200b", "кіт\u200b", "кіт"),
    ],
)
def test_normalized_forms_match_originals(
    artifact_config: PipelineConfig,
    artifact_sources: tuple[InputIdentity, ...],
    pseudo: bool,
    original: str,
    normalized: str,
    bad: str,
) -> None:
    source = artifact_sources[-1 if pseudo else 0]
    value = sentence(source, pseudo=pseudo) | {
        "original_forms": [original, "_"],
        "normalized_forms": [normalized, "_"],
        "token_spans": [None, None],
    }
    record = parse_sentence_record(value, config=artifact_config, source=source)
    assert record.original_forms == (original, "_")
    assert record.normalized_forms == (normalized, "_")
    with pytest.raises(ValidationError, match="normalized_forms"):
        parse_sentence_record(
            value | {"normalized_forms": [bad, "_"]},
            config=artifact_config,
            source=source,
        )
