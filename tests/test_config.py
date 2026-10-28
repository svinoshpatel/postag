"""Configuration checks use temporary repositories and opaque, non-corpus bytes."""

import copy
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from crf.utils.config import ConfigError, load_config, parse_config
from crf.utils.paths import PathValidationError, resolve_cli_path, validate_output_path


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    gold = tmp_path / "data/raw/gold"
    gold.mkdir(parents=True)
    (tmp_path / "data/raw/literature").mkdir()
    for name in ("train", "val", "test"):
        (gold / f"{name}.conllu").write_bytes(b"\xff opaque, not annotations")
    (tmp_path / "config").mkdir()
    return tmp_path


def minimal() -> dict[str, object]:
    return {
        "config_version": 1,
        "data": {
            "train_gold": "data/raw/gold/train.conllu",
            "val_gold": "data/raw/gold/val.conllu",
            "test_gold": "data/raw/gold/test.conllu",
            "ud_release": "fixture-release",
        },
    }


def write_config(root: Path, value: object) -> Path:
    path = root / "config/settings.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_defaults_round_trip_and_immutable(repository: Path) -> None:
    write_config(repository, minimal())
    config = load_config("config/settings.yaml", repository_root=repository)
    effective = config.to_dict()
    assert effective == {
        "config_version": 1,
        "seed": 42,
        "tag_mapping": {},
        "data": minimal()["data"]
        | {"literature_dir": "data/raw/literature", "source_encoding": "utf-8"},
        "preprocessing": {"normalizer_version": 1, "tokenizer_adapter_version": 1},
        "features": {"schema_version": 1},
        "training": {
            "c1_grid": [0.01, 0.1, 1.0],
            "c2_grid": [0.01, 0.1, 1.0],
            "parameters": {
                "max_iterations": 200,
                "num_memories": 6,
                "epsilon": 1e-5,
                "period": 10,
                "delta": 1e-5,
                "min_freq": 0.0,
                "all_possible_states": False,
                "all_possible_transitions": True,
            },
        },
        "self_training": {
            "thresholds": [0.90, 0.95, 0.98],
            "pseudo_to_gold_ratio": 1.0,
        },
        "resources": {
            "read_buffer_bytes": 65536,
            "max_paragraph_chars": 65536,
            "max_sentence_chars": 16384,
            "max_sentence_tokens": 256,
            "batch_tokens": 4096,
            "memory_budget_mib": 8192,
        },
        "evaluation": {"bootstrap_resamples": 2000, "bootstrap_seed": 42},
    }
    assert (
        len(config.training.c1_grid)
        * len(config.training.c2_grid)
        * (1 + len(config.self_training.thresholds))
        == 36
    )
    assert (
        parse_config(json.loads(json.dumps(effective)), repository_root=repository)
        == config
    )
    with pytest.raises(FrozenInstanceError):
        config.seed = 9
    effective["seed"] = 123
    assert config.seed == 42


def test_overrides_boundaries_and_seed_precedence(repository: Path) -> None:
    value = minimal() | {
        "seed": 7,
        "tag_mapping": {key: "FUNC" for key in ("ADP", "CCONJ", "SCONJ", "PART")},
        "training": {
            "c1_grid": [0],
            "c2_grid": [2],
            "parameters": {
                "max_iterations": 0,
                "epsilon": 0,
                "delta": 0,
                "min_freq": 0.5,
            },
        },
        "self_training": {"thresholds": [0, 1], "pseudo_to_gold_ratio": 0},
        "evaluation": {"bootstrap_seed": 99, "bootstrap_resamples": 1},
    }
    path = write_config(repository, value)
    assert load_config(path, repository_root=repository).seed == 7
    config = load_config(path, repository_root=repository, seed=0)
    assert config.seed == 0
    assert config.evaluation.bootstrap_seed == 99
    assert config.self_training.thresholds == (0.0, 1.0)
    assert config.training.parameters.max_iterations == 0
    assert dict(config.tag_mapping)["PART"] == "FUNC"
    assert config.training.c1_grid == (0.0,)
    assert resolve_cli_path("config/settings.yaml", repository) == path
    assert resolve_cli_path(path, repository) == path
    with pytest.raises(ConfigError, match="CLI seed"):
        load_config(path, repository_root=repository, seed=True)


@pytest.mark.parametrize(
    "section,field,bad",
    [
        (None, "config_version", 2),
        (None, "config_version", True),
        (None, "seed", -1),
        (None, "seed", "42"),
        (None, "seed", True),
        (None, "extra", 1),
        (None, "data", None),
        ("data", "ud_release", " "),
        ("data", "source_encoding", "no-such-codec"),
        ("data", "source_encoding", "base64_codec"),
        ("data", "extra", 1),
        ("preprocessing", "normalizer_version", 2),
        ("preprocessing", "tokenizer_adapter_version", True),
        ("features", "schema_version", "1"),
        ("features", "extra", 1),
        ("training", "c1_grid", []),
        ("training", "c2_grid", [0, 0.0]),
        ("training", "c1_grid", [True]),
        ("training", "c1_grid", [-1]),
        ("training", "c1_grid", [float("inf")]),
        ("training", "c1_grid", [float("nan")]),
        ("training", "c1_grid", ["0.1"]),
        ("training", "algorithm", "ap"),
        ("training.parameters", "max_iterations", -1),
        ("training.parameters", "num_memories", 0),
        ("training.parameters", "period", 0),
        ("training.parameters", "epsilon", -1),
        ("training.parameters", "delta", float("nan")),
        ("training.parameters", "min_freq", True),
        ("training.parameters", "all_possible_states", 0),
        ("training.parameters", "all_possible_transitions", "true"),
        ("training.parameters", "random_state", 42),
        ("self_training", "thresholds", [1.01]),
        ("self_training", "thresholds", [0.9, 0.9]),
        ("self_training", "pseudo_to_gold_ratio", -0.1),
        ("resources", "batch_tokens", 255),
        ("resources", "memory_budget_mib", 0),
        ("resources", "read_buffer_bytes", True),
        ("resources", "max_sentence_chars", 1.0),
        ("evaluation", "bootstrap_resamples", 0),
        ("evaluation", "bootstrap_seed", -1),
        ("evaluation", "extra", 1),
    ],
)
def test_invalid_fields(
    repository: Path, section: str | None, field: str, bad: object
) -> None:
    value = copy.deepcopy(minimal())
    target = value
    if section:
        for key in section.split("."):
            target = target.setdefault(key, {})
    target[field] = bad
    with pytest.raises(ConfigError):
        load_config(write_config(repository, value), repository_root=repository)


@pytest.mark.parametrize(
    "mapping",
    [
        {"ADP": "FUNC"},
        {key: "NOUN" for key in ("ADP", "CCONJ", "SCONJ", "PART")},
        {"ADP": "FUNC", "CCONJ": "FUNC", "SCONJ": "FUNC", "PART": "OTHER"},
        {key: "two words" for key in ("ADP", "CCONJ", "SCONJ", "PART")},
        {key: "" for key in ("ADP", "CCONJ", "SCONJ", "PART")},
        {"NOUN": "FUNC"},
    ],
)
def test_invalid_tag_mapping(repository: Path, mapping: object) -> None:
    with pytest.raises(ConfigError, match="tag_mapping"):
        load_config(
            write_config(repository, minimal() | {"tag_mapping": mapping}),
            repository_root=repository,
        )


@pytest.mark.parametrize(
    "text",
    [
        "config_version: 1\nconfig_version: 1",
        "data: {train_gold: a, train_gold: b}",
        "training: {parameters: {epsilon: 0, epsilon: 1}}",
        "1: value",
        "[]",
        "",
        "!!python/object:builtins.object {}",
        "data: &data {self: *data}",
        "data: {<<: {ud_release: release}}",
        "data: [",
        "config_version: 1\n---\n{}",
    ],
)
def test_bad_yaml(repository: Path, text: str) -> None:
    path = repository / "config/settings.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="settings.yaml"):
        load_config(path, repository_root=repository)


@pytest.mark.parametrize(
    "kind",
    [
        "same",
        "hardlink",
        "symlink",
        "outside",
        "missing",
        "extension",
        "literature",
        "literature_symlink",
    ],
)
def test_source_path_failures(repository: Path, tmp_path: Path, kind: str) -> None:
    value = minimal()
    data = value["data"]
    gold = repository / "data/raw/gold"
    if kind == "same":
        data["val_gold"] = data["train_gold"]
    elif kind in {"hardlink", "symlink"}:
        alias = gold / "alias.conllu"
        if kind == "hardlink":
            os.link(gold / "train.conllu", alias)
        else:
            alias.symlink_to(gold / "train.conllu")
        data["val_gold"] = "data/raw/gold/alias.conllu"
    elif kind == "outside":
        (gold / "escape.conllu").symlink_to(repository.parent / "external.conllu")
        data["val_gold"] = "data/raw/gold/escape.conllu"
    elif kind == "missing":
        data["val_gold"] = "data/raw/gold/absent.conllu"
    elif kind == "extension":
        data["val_gold"] = "data/raw/gold/val.txt"
    elif kind == "literature":
        data["literature_dir"] = "data/raw/gold"
    else:
        (repository / "alias").symlink_to(
            repository / "data/raw/literature", target_is_directory=True
        )
        data["literature_dir"] = "alias"
    with pytest.raises(ConfigError):
        load_config(write_config(repository, value), repository_root=repository)


def test_metadata_parse_without_originals_and_missing_literature(
    tmp_path: Path, repository: Path
) -> None:
    assert (
        parse_config(minimal(), repository_root=tmp_path / "absent").data.ud_release
        == "fixture-release"
    )
    (repository / "data/raw/literature").rmdir()
    with pytest.raises(ConfigError, match="Missing literature"):
        load_config(write_config(repository, minimal()), repository_root=repository)


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/gold/new.conllu",
        "../escape",
        "reports",
        "other/new",
        "reports/../../data/raw/new",
    ],
)
def test_output_rejections(repository: Path, path: str) -> None:
    with pytest.raises(PathValidationError):
        validate_output_path(path, repository)


def test_fresh_existing_and_redirected_outputs(repository: Path) -> None:
    for tree in ("data/interim", "data/processed", "models", "reports"):
        candidate = validate_output_path(f"{tree}/fresh/output.json", repository)
        assert candidate == repository / tree / "fresh/output.json"
        assert not candidate.parent.exists()
    reports = repository / "reports"
    reports.mkdir()
    existing = reports / "existing"
    existing.write_text("keep", encoding="utf-8")
    (reports / "dangling").symlink_to(reports / "absent")
    (reports / "raw").symlink_to(repository / "data/raw")
    for path in (existing, reports / "dangling", reports / "raw/new.json"):
        with pytest.raises(PathValidationError):
            validate_output_path(path, repository)
    assert existing.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "section",
    [
        "preprocessing",
        "features",
        "training",
        "self_training",
        "resources",
        "evaluation",
    ],
)
def test_sections_must_be_mappings(repository: Path, section: str) -> None:
    with pytest.raises(ConfigError, match=section):
        load_config(
            write_config(repository, minimal() | {section: []}),
            repository_root=repository,
        )


@pytest.mark.parametrize("key", ["train_gold", "val_gold", "test_gold", "ud_release"])
def test_required_data_fields(repository: Path, key: str) -> None:
    value = minimal()
    del value["data"][key]
    with pytest.raises(ConfigError, match=key):
        load_config(write_config(repository, value), repository_root=repository)


def test_all_sections_allow_explicit_overrides(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = minimal() | {
        "preprocessing": {"normalizer_version": 1, "tokenizer_adapter_version": 1},
        "features": {"schema_version": 1},
        "training": {
            "c1_grid": [0.2],
            "c2_grid": [0.3],
            "parameters": {
                "max_iterations": 9,
                "num_memories": 3,
                "epsilon": 0.5,
                "period": 2,
                "delta": 0.6,
                "min_freq": 0.7,
                "all_possible_states": True,
                "all_possible_transitions": False,
            },
        },
        "resources": {
            "read_buffer_bytes": 1,
            "max_paragraph_chars": 2,
            "max_sentence_chars": 3,
            "max_sentence_tokens": 4,
            "batch_tokens": 4,
            "memory_budget_mib": 5,
        },
    }
    value["data"]["source_encoding"] = "cp1251"
    path = write_config(repository, value)
    monkeypatch.chdir(repository / "config")
    config = load_config("config/settings.yaml", repository_root=repository)
    assert config.to_dict()["training"] == value["training"]
    assert config.to_dict()["resources"] == value["resources"]
    assert config.data.source_encoding == "cp1251"
    # Nonrecursive YAML aliases contain plain values and remain supported.
    text = path.read_text(encoding="utf-8")
    text += "\nseed: &seed 7\nevaluation: {bootstrap_seed: *seed}\n"
    path.write_text(text, encoding="utf-8")
    assert load_config(path, repository_root=repository).evaluation.bootstrap_seed == 7


def test_missing_and_non_utf8_config(repository: Path) -> None:
    with pytest.raises(ConfigError, match="missing.yaml"):
        load_config("missing.yaml", repository_root=repository)
    path = repository / "invalid.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError, match="invalid.yaml"):
        load_config(path, repository_root=repository)
