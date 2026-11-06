"""Shared strict observation headers for inference/evaluation completion."""

import math
import platform
import re
import unicodedata
from datetime import datetime
from typing import cast

from crf.models.training import dependency_versions
from crf.utils.serialization import ValidationError, object_fields, text_value


def validate_observation(
    value: object, started_at: object, finished_at: object, peak: object
) -> None:
    raw = object_fields(
        value,
        {
            "python_version",
            "unicode_version",
            "dependency_versions",
            "source_revision",
            "source_dirty",
        },
        "environment",
    )
    if (
        raw["python_version"] != platform.python_version()
        or raw["unicode_version"] != unicodedata.unidata_version
        or raw["dependency_versions"] != dict(dependency_versions())
    ):
        raise ValidationError("Recorded inference environment is incompatible")
    revision = raw["source_revision"]
    if revision is not None and (
        type(revision) is not str
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)
    ):
        raise ValidationError("Invalid source revision")
    if raw["source_dirty"] is not None and type(raw["source_dirty"]) is not bool:
        raise ValidationError("Invalid source dirty state")
    times = []
    for timestamp in (started_at, finished_at):
        text = text_value(timestamp, "timestamp")
        if not text.endswith("Z"):
            raise ValidationError("Timestamp must use UTC Z")
        times.append(datetime.fromisoformat(text))
    if times[1] < times[0]:
        raise ValidationError("Finish precedes start")
    try:
        valid_peak = (
            type(peak) in (int, float)
            and math.isfinite(cast(float, peak))
            and cast(float, peak) >= 0
        )
    except OverflowError:
        valid_peak = False
    if not valid_peak:
        raise ValidationError("Invalid peak RSS")
