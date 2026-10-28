"""Strict version-1 pipeline settings, safe YAML loading, and input validation."""

import codecs
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml
from yaml.nodes import MappingNode

from crf.utils.paths import (
    PathValidationError,
    reject_symlink_ancestors,
    repository_path,
    resolve_cli_path,
)

UPOS = frozenset(
    "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()
)
SERVICE_TAGS = frozenset({"ADP", "CCONJ", "SCONJ", "PART"})


class ConfigError(ValueError):
    """Configuration syntax, schema, or settings are invalid."""


class _StrictLoader(yaml.SafeLoader):
    """Local safe loader with duplicate detection and no merge-key expansion."""


def _construct_mapping(loader: yaml.SafeLoader, node: MappingNode) -> dict[str, object]:
    result: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if type(key) is not str:
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_StrictLoader.add_constructor("tag:yaml.org,2002:map", _construct_mapping)


@dataclass(frozen=True)
class DataConfig:
    train_gold: Path
    val_gold: Path
    test_gold: Path
    ud_release: str
    literature_dir: Path = Path("data/raw/literature")
    source_encoding: str = "utf-8"


@dataclass(frozen=True)
class PreprocessingConfig:
    normalizer_version: int = 1
    tokenizer_adapter_version: int = 1


@dataclass(frozen=True)
class FeatureConfig:
    schema_version: int = 1


@dataclass(frozen=True)
class CRFParameters:
    max_iterations: int = 200
    num_memories: int = 6
    epsilon: float = 1e-5
    period: int = 10
    delta: float = 1e-5
    min_freq: float = 0.0
    all_possible_states: bool = False
    all_possible_transitions: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    c1_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    c2_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    parameters: CRFParameters = CRFParameters()


@dataclass(frozen=True)
class SelfTrainingConfig:
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.98)
    pseudo_to_gold_ratio: float = 1.0


@dataclass(frozen=True)
class ResourceConfig:
    read_buffer_bytes: int = 65536
    max_paragraph_chars: int = 65536
    max_sentence_chars: int = 16384
    max_sentence_tokens: int = 256
    batch_tokens: int = 4096
    memory_budget_mib: int = 8192


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    config_version: int = 1
    seed: int = 42
    tag_mapping: tuple[tuple[str, str], ...] = ()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    features: FeatureConfig = FeatureConfig()
    training: TrainingConfig = TrainingConfig()
    self_training: SelfTrainingConfig = SelfTrainingConfig()
    resources: ResourceConfig = ResourceConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def to_dict(self) -> dict[str, object]:
        """Return complete JSON-compatible effective settings without machine paths."""
        result = asdict(self)
        result["tag_mapping"] = dict(self.tag_mapping)
        for key in ("train_gold", "val_gold", "test_gold", "literature_dir"):
            result["data"][key] = getattr(self.data, key).as_posix()
        for key in ("c1_grid", "c2_grid"):
            result["training"][key] = list(getattr(self.training, key))
        result["self_training"]["thresholds"] = list(self.self_training.thresholds)
        return result


def _mapping(value: object, field: str, allowed: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ConfigError(f"{field}: expected a mapping with string keys")
    result = cast(dict[str, object], value)
    unknown = result.keys() - allowed
    if unknown:
        raise ConfigError(f"{field}: unknown fields {sorted(unknown)}")
    return result


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field}: expected integer >= {minimum}")
    return value


def _number(value: object, field: str, maximum: float = math.inf) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{field}: expected a finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise ConfigError(f"{field}: number is too large") from exc
    if not math.isfinite(result) or not 0 <= result <= maximum:
        raise ConfigError(f"{field}: expected finite number in [0, {maximum}]")
    return result


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{field}: expected nonempty string without NUL")
    return value


def _grid(value: object, field: str, maximum: float = math.inf) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field}: expected nonempty array")
    result = tuple(_number(item, field, maximum) for item in value)
    if len(set(result)) != len(result):
        raise ConfigError(f"{field}: duplicate entries")
    return result


def _section(
    raw: dict[str, object], name: str, defaults: dict[str, object]
) -> dict[str, object]:
    fields = {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in defaults.items()
    }
    supplied = _mapping(raw.get(name, {}), name, set(fields))
    return fields | supplied


def parse_config(
    value: object, *, repository_root: Path, seed: int | None = None
) -> PipelineConfig:
    """Validate settings without opening corpora (also suitable for model metadata)."""
    raw = _mapping(
        value,
        "config",
        {
            "config_version",
            "seed",
            "tag_mapping",
            "data",
            "preprocessing",
            "features",
            "training",
            "self_training",
            "resources",
            "evaluation",
        },
    )
    if _integer(raw.get("config_version"), "config_version") != 1:
        raise ConfigError("Unsupported config_version")
    configured_seed = _integer(raw.get("seed", 42), "seed")
    effective_seed = configured_seed if seed is None else _integer(seed, "CLI seed")
    data = _mapping(
        raw.get("data"),
        "data",
        {
            "train_gold",
            "val_gold",
            "test_gold",
            "ud_release",
            "literature_dir",
            "source_encoding",
        },
    )
    paths: dict[str, Path] = {}
    for key in ("train_gold", "val_gold", "test_gold", "literature_dir"):
        name = _string(
            data.get(key, "data/raw/literature" if key == "literature_dir" else None),
            f"data.{key}",
        )
        # Check lexical literature symlinks before canonicalizing the path.
        if key == "literature_dir":
            reject_symlink_ancestors(
                repository_root.resolve() / name, repository_root.resolve()
            )
        paths[key] = repository_path(name, repository_root, f"data.{key}")
        if key != "literature_dir" and (
            not paths[key].is_relative_to("data/raw/gold")
            or paths[key].suffix != ".conllu"
        ):
            raise ConfigError(f"data.{key}: expected .conllu path under data/raw/gold")
    if paths["literature_dir"] != Path("data/raw/literature"):
        raise ConfigError("data.literature_dir: must be data/raw/literature")
    if len({paths[key] for key in ("train_gold", "val_gold", "test_gold")}) != 3:
        raise ConfigError("Gold split paths must be distinct")
    encoding = _string(data.get("source_encoding", "utf-8"), "data.source_encoding")
    try:
        (
            codecs.getincrementaldecoder(encoding)(errors="strict").decode(
                b"", final=True
            )
            + ""
        )
    except (LookupError, TypeError, ValueError) as exc:
        raise ConfigError(
            f"data.source_encoding: unsupported text codec {encoding}"
        ) from exc
    mapping = _mapping(raw.get("tag_mapping", {}), "tag_mapping", set(SERVICE_TAGS))
    if mapping:
        targets = [_string(item, "tag_mapping target") for item in mapping.values()]
        if (
            mapping.keys() != SERVICE_TAGS
            or len(set(targets)) != 1
            or targets[0] in UPOS
            or any(char.isspace() for char in targets[0])
        ):
            raise ConfigError(
                "tag_mapping: map all four service tags to one new whitespace-free label"
            )
    versions = _section(raw, "preprocessing", asdict(PreprocessingConfig()))
    features = _section(raw, "features", asdict(FeatureConfig()))
    for key, version in (versions | features).items():
        if _integer(version, key) != 1:
            raise ConfigError(f"Unsupported {key}: {version}")
    training = _section(raw, "training", asdict(TrainingConfig()))
    params = _mapping(
        training["parameters"], "training.parameters", set(asdict(CRFParameters()))
    )
    parameters = asdict(CRFParameters()) | params
    for key in ("max_iterations", "num_memories", "period"):
        parameters[key] = _integer(
            parameters[key], key, 0 if key == "max_iterations" else 1
        )
    for key in ("epsilon", "delta", "min_freq"):
        parameters[key] = _number(parameters[key], key)
    for key in ("all_possible_states", "all_possible_transitions"):
        if type(parameters[key]) is not bool:
            raise ConfigError(f"{key}: expected boolean")
    self_training = _section(raw, "self_training", asdict(SelfTrainingConfig()))
    resources = {
        key: _integer(item, f"resources.{key}", 1)
        for key, item in _section(raw, "resources", asdict(ResourceConfig())).items()
    }
    if resources["batch_tokens"] < resources["max_sentence_tokens"]:
        raise ConfigError("resources.batch_tokens must be >= max_sentence_tokens")
    evaluation = _section(raw, "evaluation", asdict(EvaluationConfig()))
    return PipelineConfig(
        data=DataConfig(
            **paths,
            ud_release=_string(data.get("ud_release"), "data.ud_release"),
            source_encoding=encoding,
        ),
        seed=effective_seed,
        tag_mapping=tuple(
            sorted((key, cast(str, item)) for key, item in mapping.items())
        ),
        preprocessing=PreprocessingConfig(
            normalizer_version=_integer(
                versions["normalizer_version"], "normalizer_version"
            ),
            tokenizer_adapter_version=_integer(
                versions["tokenizer_adapter_version"], "tokenizer_adapter_version"
            ),
        ),
        features=FeatureConfig(
            schema_version=_integer(features["schema_version"], "schema_version")
        ),
        training=TrainingConfig(
            _grid(
                training["c1_grid"],
                "training.c1_grid",
            ),
            _grid(
                training["c2_grid"],
                "training.c2_grid",
            ),
            CRFParameters(**parameters),
        ),
        self_training=SelfTrainingConfig(
            _grid(
                self_training["thresholds"],
                "self_training.thresholds",
                1,
            ),
            _number(
                self_training["pseudo_to_gold_ratio"],
                "self_training.pseudo_to_gold_ratio",
            ),
        ),
        resources=ResourceConfig(**resources),
        evaluation=EvaluationConfig(
            _integer(
                evaluation["bootstrap_resamples"], "evaluation.bootstrap_resamples", 1
            ),
            _integer(evaluation["bootstrap_seed"], "evaluation.bootstrap_seed"),
        ),
    )


def validate_input_paths(config: PipelineConfig, repository_root: Path) -> None:
    """Check source locations and physical identities without reading annotations."""
    root = repository_root.resolve()
    gold_root = (root / "data/raw/gold").resolve()
    literature = root / config.data.literature_dir
    reject_symlink_ancestors(literature, root)
    if (
        not gold_root.is_relative_to(root)
        or not literature.resolve().is_relative_to(root)
        or literature.resolve().is_relative_to(gold_root)
    ):
        raise PathValidationError("Gold/literature source roots escape or overlap")
    identities: set[tuple[int, int]] = set()
    for key in ("train_gold", "val_gold", "test_gold"):
        path = (root / getattr(config.data, key)).resolve()
        if (
            not path.is_relative_to(gold_root)
            or path.suffix != ".conllu"
            or not path.is_file()
        ):
            raise PathValidationError(
                f"data.{key}: missing or invalid gold file: {path}"
            )
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise PathValidationError(
                f"data.{key}: gold split aliases another physical file"
            )
        identities.add(identity)
    if not literature.is_dir():
        raise PathValidationError(f"Missing literature directory: {literature}")


def load_config(
    path: str | Path, *, repository_root: Path, seed: int | None = None
) -> PipelineConfig:
    """Load safe UTF-8 YAML, apply defaults/seed override, and validate sources."""
    source = resolve_cli_path(path, repository_root)
    try:
        with source.open(encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_StrictLoader)
        result = parse_config(value, repository_root=repository_root, seed=seed)
        validate_input_paths(result, repository_root)
        return result
    except (yaml.YAMLError, OSError, UnicodeError, RecursionError, ValueError) as exc:
        raise ConfigError(f"{source}: {exc}") from exc
