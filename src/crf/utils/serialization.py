"""Strict JSON primitives and incremental JSONL reading; no output publication."""

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import cast

from crf.utils.config import PipelineConfig


class ValidationError(ValueError):
    """An artifact violates its declared syntax, schema, or integrity contract."""


def object_fields(value: object, fields: set[str], context: str) -> dict[str, object]:
    """Require an object with exactly the declared string keys."""
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValidationError(f"{context}: expected an object with string keys")
    result = cast(dict[str, object], value)
    missing, extra = fields - result.keys(), result.keys() - fields
    if missing or extra:
        raise ValidationError(
            f"{context}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}"
        )
    return result


def integer(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{field}: expected integer >= {minimum}")
    return value


def text_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field}: expected nonempty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValidationError(f"{field}: invalid Unicode") from exc
    return value


def sha256_value(value: object, field: str) -> str:
    result = text_value(value, field)
    if not re.fullmatch("[0-9a-f]{64}", result):
        raise ValidationError(f"{field}: expected lowercase SHA-256")
    return result


def source_path_value(value: object) -> str:
    path = text_value(value, "source_path")
    if (
        "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or PurePosixPath(path).is_absolute()
    ):
        raise ValidationError(
            "source_path: expected repository-relative POSIX path without dot components"
        )
    return path


def probability(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValidationError(f"{field}: expected probability")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValidationError(f"{field}: invalid probability") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValidationError(f"{field}: expected finite probability in [0, 1]")
    return result


def array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValidationError(f"{field}: expected array")
    return value


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValidationError(f"Nonfinite JSON number: {value}")


def _check_json(value: object) -> None:
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for key, item in value.items():
            _check_json(key)
            _check_json(item)
        return
    raise ValidationError("Expected plain JSON values with finite numbers")


def loads_json(value: str | bytes) -> object:
    """Decode UTF-8 JSON without duplicates, BOMs, surrogates, or nonfinite values."""
    try:
        decoded = (
            value.decode("utf-8", errors="strict")
            if isinstance(value, bytes)
            else value
        )
        if decoded.startswith("\ufeff"):
            raise ValidationError("JSON must not contain a BOM")
        result: object = json.loads(
            decoded, object_pairs_hook=_pairs, parse_constant=_constant
        )
        _check_json(result)
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError(str(exc)) from exc


def json_bytes(value: object, *, sort_keys: bool = False) -> bytes:
    """Encode compact UTF-8 JSON, strictly rejecting non-JSON or nonfinite values."""
    try:
        _check_json(value)
        return json.dumps(
            value,
            sort_keys=sort_keys,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError(str(exc)) from exc


def jsonl_bytes(value: object) -> bytes:
    """Encode one JSON object and its LF; output allocation belongs to the caller."""
    if not isinstance(value, dict):
        raise ValidationError("JSONL record must be an object")
    return json_bytes(value) + b"\n"


def read_json(path: Path) -> object:
    try:
        return loads_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ValidationError(f"{path}: {exc}") from exc


def iter_jsonl[T](path: Path, validator: Callable[[object], T]) -> Iterator[T]:
    """Read and validate one object per line without collecting the whole file."""
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    if b"\r" in line:
                        raise ValidationError("JSONL requires LF line endings")
                    value = loads_json(line)
                    if not isinstance(value, dict):
                        raise ValidationError("JSONL record must be an object")
                    record = validator(value)
                except ValueError as exc:
                    raise ValidationError(
                        f"{path}: record/line {line_number}: {exc}"
                    ) from exc
                yield record
    except OSError as exc:
        raise ValidationError(f"{path}: {exc}") from exc


def canonical_config_bytes(config: PipelineConfig) -> bytes:
    """Encode the complete effective configuration under frozen-selection rules."""
    return json_bytes(config.to_dict(), sort_keys=True)


def config_sha256(config: PipelineConfig) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()
