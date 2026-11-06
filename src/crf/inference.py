"""Independent bounded raw inference from a trusted standalone model bundle."""

import codecs
import hashlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO

from crf.data.identity import digest_file, sentence_id
from crf.data.literature import TextExclusion, iter_text_events
from crf.evaluation import ModelReference
from crf.experiments import file_sha256, read_bytes
from crf.models.bundles import load_bundle, parse_metadata
from crf.models.confidence import score_sentence
from crf.pipeline import _context, _publish, _utc_now
from crf.predictions import Prediction, parse_prediction
from crf.reporting import validate_failure, validate_observation
from crf.utils.artifacts import allocate_run, atomic_writer, publish_completion
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    integer,
    json_bytes,
    jsonl_bytes,
    loads_json,
    object_fields,
)


@dataclass(frozen=True)
class RawInput:
    source_path: str
    source_sha256: str
    size_bytes: int
    encoding: str

    def to_dict(self) -> dict[str, object]:
        return dict(source_role="raw", **asdict(self))


@dataclass
class StreamState:
    """Mutable because decoded lengths/exclusions are collected incrementally."""

    decoded_characters: int = 0
    exclusions: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class PredictionRun:
    run_id: str
    predictions_path: Path
    completion_path: Path
    sentence_count: int
    token_count: int
    label_space: str


def _chunks(
    path: Path,
    source: RawInput,
    config: PipelineConfig,
    monitor: MemoryMonitor,
    state: StreamState,
) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder(source.encoding)(errors="strict")
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(config.resources.read_buffer_bytes)
            monitor.checkpoint()
            pending = len(decoder.getstate()[0])
            try:
                decoded = decoder.decode(chunk, final=not chunk)
            except UnicodeError as exc:
                raise ValidationError(
                    f"{path}: {source.encoding} decoding error near byte {consumed - pending + getattr(exc, 'start', 0)}"
                ) from exc
            digest.update(chunk)
            consumed += len(chunk)
            state.decoded_characters += len(decoded)
            if decoded:
                yield decoded
            if not chunk:
                break
    if (digest.hexdigest(), consumed) != (source.source_sha256, source.size_bytes):
        raise ValidationError("Raw input changed during decoding")
    actual = digest_file(
        path, buffer_bytes=config.resources.read_buffer_bytes, monitor=monitor
    )
    if (actual.sha256, actual.size_bytes) != (source.source_sha256, source.size_bytes):
        raise ValidationError("Raw input changed after decoding")


def _check_predictions(
    stream: BinaryIO,
    config: PipelineConfig,
    reference: ModelReference,
    source: RawInput,
    run_id: str,
    decoded_length: int,
) -> tuple[int, int]:
    sentences = tokens = previous = 0
    for line in stream:
        prediction = parse_prediction(loads_json(line), config)
        if (
            prediction.run_id,
            prediction.model_ref,
            prediction.source_path,
            prediction.source_sha256,
            prediction.source_role,
        ) != (run_id, reference, source.source_path, source.source_sha256, "raw"):
            raise ValidationError("Standalone prediction provenance mismatch")
        if prediction.sentence_ordinal <= previous or any(
            span is None or span[1] > decoded_length for span in prediction.token_spans
        ):
            raise ValidationError("Standalone prediction order/span mismatch")
        previous = prediction.sentence_ordinal
        sentences += 1
        tokens += len(prediction.token_ids)
    return sentences, tokens


def predict(
    bundle_directory: Path, input_path: Path, repository_root: Path
) -> PredictionRun:
    root = repository_root.resolve()
    directory = (root / bundle_directory).absolute()
    encoded = read_bytes(directory / "metadata.json")
    metadata = parse_metadata(loads_json(encoded))
    config = metadata.config
    reference = ModelReference(
        metadata.candidate_id,
        metadata.model_sha256,
        hashlib.sha256(encoded).hexdigest(),
    )
    path = (root / input_path).resolve(strict=True)
    if not path.is_file():
        raise ValidationError("Standalone input must be a regular file")
    recorded = (
        path.relative_to(root).as_posix()
        if path.is_relative_to(root)
        else path.as_posix()
    )
    paths = allocate_run(root, kind="predict")
    context = _context(root)
    monitor = MemoryMonitor(config.resources.memory_budget_mib)
    source: RawInput | None = None
    state = StreamState()
    sentences = tokens = 0
    validated_predictions = False
    output = paths.reports / "predictions.jsonl"
    completion = paths.reports / "completion.json"
    label_space = "mapped_target_labels" if config.tag_mapping else "UD_UPOS"

    def descriptor(error: BaseException | None = None) -> dict[str, object]:
        assert source is not None
        return dict(
            schema_version=1,
            run_id=paths.run_id,
            status="complete" if error is None else "failed",
            started_at=context.started_at,
            finished_at=_utc_now(),
            model_ref=reference.to_dict(),
            input=source.to_dict(),
            predictions_sha256=file_sha256(output) if validated_predictions else None,
            sentence_count=sentences if validated_predictions else 0,
            token_count=tokens if validated_predictions else 0,
            environment=dict(
                python_version=context.python_version,
                unicode_version=context.unicode_version,
                dependency_versions=dict(context.dependency_versions),
                source_revision=context.source_revision,
                source_dirty=context.source_dirty,
            ),
            peak_rss_mib=monitor.peak_rss_mib,
            error=None
            if error is None
            else dict(stage="prediction", message=str(error) or type(error).__name__),
        )

    try:
        with monitor:
            digest = digest_file(
                path, buffer_bytes=config.resources.read_buffer_bytes, monitor=monitor
            )
            source = RawInput(
                recorded, digest.sha256, digest.size_bytes, config.data.source_encoding
            )
            loaded = load_bundle(
                directory,
                expected_model_sha256=reference.model_sha256,
                expected_metadata_sha256=reference.metadata_sha256,
                monitor=monitor,
            )
            try:

                def validate(stream: BinaryIO) -> None:
                    if _check_predictions(
                        stream,
                        config,
                        reference,
                        source,
                        paths.run_id,
                        state.decoded_characters,
                    ) != (sentences, tokens):
                        raise ValidationError("Standalone prediction count mismatch")

                with atomic_writer(output, root, validate=validate) as stream:
                    for event in iter_text_events(
                        _chunks(path, source, config, monitor, state),
                        config=config,
                        source_label=recorded,
                        monitor=monitor,
                    ):
                        if isinstance(event, TextExclusion):
                            state.exclusions[event.reason] += 1
                            continue
                        text = event.text
                        confidence = score_sentence(
                            text.normalized_forms,
                            model=loaded.model,
                            config=config,
                            monitor=monitor,
                        )
                        prediction = Prediction(
                            1,
                            paths.run_id,
                            reference,
                            sentence_id(
                                "raw",
                                recorded,
                                source.source_sha256,
                                event.sentence_ordinal,
                            ),
                            recorded,
                            source.source_sha256,
                            "raw",
                            event.sentence_ordinal,
                            tuple(range(1, len(text.original_forms) + 1)),
                            text.original_forms,
                            text.normalized_forms,
                            text.token_spans,
                            confidence.labels,
                            confidence.token_confidences,
                        )
                        stream.write(jsonl_bytes(prediction.to_dict()))
                        sentences += 1
                        tokens += len(text.original_forms)
                validated_predictions = True
                _publish(
                    paths.reports / "diagnostics.json",
                    json_bytes(
                        dict(
                            schema_version=1,
                            run_id=paths.run_id,
                            resource_exclusions=dict(state.exclusions),
                            decoded_characters=state.decoded_characters,
                            target_inventory=list(config.target_inventory),
                            label_space=label_space,
                            limitations=[
                                "Raw tokenizer/UD multiword-token compatibility remains limited; no corpus overlap, MWT, deduplication or confidence selection is applied"
                            ],
                        )
                    ),
                    root,
                )
                if (
                    file_sha256(directory / "metadata.json", monitor)
                    != reference.metadata_sha256
                    or file_sha256(directory / "model.joblib", monitor)
                    != reference.model_sha256
                ):
                    raise ValidationError("Standalone bundle changed during inference")
            finally:
                loaded.close()
        monitor.ensure_success()
    except BaseException as exc:
        if source is not None:
            publish_completion(
                completion,
                descriptor(exc),
                root,
                validate=validate_failure,
                monitor=monitor,
            )
        raise

    def validate_completion(value: object) -> None:
        assert source is not None
        raw = object_fields(
            value,
            {
                "schema_version",
                "run_id",
                "status",
                "started_at",
                "finished_at",
                "model_ref",
                "input",
                "predictions_sha256",
                "sentence_count",
                "token_count",
                "environment",
                "peak_rss_mib",
                "error",
            },
            "prediction completion",
        )
        if (
            integer(raw["schema_version"], "schema_version") != 1
            or raw["run_id"] != paths.run_id
            or raw["status"] != "complete"
            or raw["error"] is not None
            or raw["model_ref"] != reference.to_dict()
            or raw["input"] != source.to_dict()
        ):
            raise ValidationError("Invalid standalone completion identity")
        raw_input = object_fields(
            raw["input"],
            {"source_role", "source_path", "source_sha256", "size_bytes", "encoding"},
            "raw input",
        )
        integer(raw_input["size_bytes"], "input byte count")
        validate_observation(
            raw["environment"],
            raw["started_at"],
            raw["finished_at"],
            raw["peak_rss_mib"],
        )
        actual = digest_file(path, buffer_bytes=config.resources.read_buffer_bytes)
        if (actual.sha256, actual.size_bytes) != (
            source.source_sha256,
            source.size_bytes,
        ):
            raise ValidationError("Standalone source changed before publication")
        if raw["predictions_sha256"] != file_sha256(output):
            raise ValidationError("Standalone prediction checksum mismatch")
        from crf.models.bundles import _open_file

        with _open_file(output.parent, output.name) as stream:
            counts = _check_predictions(
                stream,
                config,
                reference,
                source,
                paths.run_id,
                state.decoded_characters,
            )
        if counts != (
            integer(raw["sentence_count"], "sentence count"),
            integer(raw["token_count"], "token count"),
        ):
            raise ValidationError("Standalone completion count mismatch")

    publish_completion(
        completion, descriptor(), root, validate=validate_completion, monitor=monitor
    )
    return PredictionRun(
        paths.run_id, output, completion, sentences, tokens, label_space
    )
