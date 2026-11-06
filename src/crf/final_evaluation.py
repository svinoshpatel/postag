"""Frozen three-role evaluation with an immutable, validated inference cache."""

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from crf.data.conllu import GoldSentence, iter_gold_records, iter_gold_sentences
from crf.data.identity import iter_input_identities
from crf.data.records import SentenceRecord
from crf.evaluation import (
    GoldVocabulary,
    ModelReference,
    SentencePrediction,
    gold_vocabulary,
    score_test,
    validate_metric_report,
)
from crf.experiments import FrozenExperiment, file_sha256, load_development, read_bytes
from crf.models.bundles import _open_file, load_bundle
from crf.models.confidence import score_sentence
from crf.pipeline import _context, _publish, _utc_now
from crf.predictions import Prediction, parse_prediction
from crf.reporting import validate_observation
from crf.utils.artifacts import _parent_fd, atomic_writer
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import (
    ValidationError,
    jsonl_bytes,
    loads_json,
    object_fields,
)

if TYPE_CHECKING:
    from crf.bootstrap import Outcome

ROLES = ("gate", "final", "gold_control")
CORE_OUTPUTS = tuple(f"{role}.jsonl" for role in ROLES) + ("metrics.json",)


@dataclass(frozen=True)
class EvaluationCore:
    experiment: FrozenExperiment
    directory: Path
    metrics: dict[str, object]
    gold: tuple[GoldSentence, ...]


def _inputs(
    experiment: FrozenExperiment, root: Path, monitor: MemoryMonitor
) -> tuple[tuple[SentenceRecord, ...], tuple[GoldSentence, ...], GoldVocabulary]:
    config = experiment.config
    inputs = tuple(iter_input_identities(config, root, monitor=monitor))
    train = tuple(iter_gold_records(inputs[0], config, root, monitor=monitor))
    gold = tuple(
        iter_gold_sentences(
            inputs[2], config, root, final_evaluation=True, monitor=monitor
        )
    )
    return (
        tuple(sentence.to_record(config, final_evaluation=True) for sentence in gold),
        gold,
        gold_vocabulary(train, config),
    )


def cached_predictions(
    path: Path,
    records: tuple[SentenceRecord, ...],
    experiment: FrozenExperiment,
    reference: ModelReference,
    monitor: MemoryMonitor,
) -> Iterator[SentencePrediction]:
    with _open_file(path.parent, path.name) as stream:
        for record, line in zip_longest(records, stream):
            monitor.checkpoint()
            if record is None or line is None:
                raise ValidationError("Missing/extra cached prediction")
            prediction = parse_prediction(loads_json(line), experiment.config)
            expected = (
                experiment.run_id,
                reference,
                record.sentence_id,
                record.source_path,
                record.source_sha256,
                "test_gold",
                record.sentence_ordinal,
                record.token_ids,
                record.original_forms,
                record.normalized_forms,
                record.token_spans,
            )
            actual = (
                prediction.run_id,
                prediction.model_ref,
                prediction.sentence_id,
                prediction.source_path,
                prediction.source_sha256,
                prediction.source_role,
                prediction.sentence_ordinal,
                prediction.token_ids,
                prediction.original_forms,
                prediction.normalized_forms,
                prediction.token_spans,
            )
            if actual != expected:
                raise ValidationError("Cached prediction alignment/provenance mismatch")
            yield SentencePrediction(
                prediction.sentence_id,
                prediction.token_ids,
                prediction.normalized_forms,
                prediction.predicted_labels,
            )


def _load_core(experiment: FrozenExperiment, root: Path) -> EvaluationCore:
    directory = root / "reports" / experiment.run_id / "evaluation"
    raw = object_fields(
        loads_json(read_bytes(directory / "core.json")),
        {
            "schema_version",
            "run_id",
            "status",
            "development_sha256",
            "selected_sha256",
            "outputs",
            "started_at",
            "finished_at",
            "environment",
            "peak_rss_mib",
        },
        "evaluation core",
    )
    if (
        raw["schema_version"] != 1
        or type(raw["schema_version"]) is not int
        or raw["run_id"] != experiment.run_id
        or raw["status"] != "predictions_complete"
    ):
        raise ValidationError("Invalid evaluation core")
    validate_observation(
        raw["environment"], raw["started_at"], raw["finished_at"], raw["peak_rss_mib"]
    )
    outputs = object_fields(raw["outputs"], set(CORE_OUTPUTS), "core outputs")
    with MemoryMonitor(experiment.config.resources.memory_budget_mib) as monitor:
        if raw["development_sha256"] != file_sha256(
            experiment.run_path, monitor
        ) or raw["selected_sha256"] != file_sha256(
            root / "models" / experiment.run_id / "selected.json", monitor
        ):
            raise ValidationError("Evaluation core frozen-history mismatch")
        for name, checksum in outputs.items():
            if checksum != file_sha256(directory / name, monitor):
                raise ValidationError("Evaluation core output checksum mismatch")
        records, gold, vocabulary = _inputs(experiment, root, monitor)
        report = validate_metric_report(
            loads_json(read_bytes(directory / "metrics.json")),
            experiment.config,
            split="test_gold",
        )
        expected_models = {}
        for role in ROLES:
            reference = getattr(experiment.selection, role)
            expected_models[role] = score_test(
                records,
                cached_predictions(
                    directory / f"{role}.jsonl", records, experiment, reference, monitor
                ),
                config=experiment.config,
                vocabulary=vocabulary,
                model_ref=reference,
                final_evaluation=True,
            )
        if report != dict(
            schema_version=1,
            run_id=experiment.run_id,
            split="test_gold",
            target_inventory=list(experiment.config.target_inventory),
            models=expected_models,
        ):
            raise ValidationError(
                "Cached test metrics disagree with aligned predictions"
            )
    monitor.ensure_success()
    return EvaluationCore(experiment, directory, report, gold)


def evaluate_core(
    root: Path, run_id: str, *, seed: int | None = None
) -> EvaluationCore:
    root = root.resolve()
    experiment = load_development(root, run_id)
    config = experiment.config
    if seed is not None and (
        type(seed) is not int or seed != config.evaluation.bootstrap_seed
    ):
        raise ValidationError("Evaluation seed must equal frozen bootstrap seed")
    directory = root / "reports" / run_id / "evaluation"
    if directory.exists() or directory.is_symlink():
        return _load_core(experiment, root)
    context = _context(root)
    with _parent_fd(directory.relative_to(root), root, create=True) as parent:
        os.mkdir(directory.name, mode=0o700, dir_fd=parent)
    with MemoryMonitor(config.resources.memory_budget_mib) as outer:
        records, gold, vocabulary = _inputs(experiment, root, outer)
        models = {}
        for role in ROLES:
            reference = getattr(experiment.selection, role)
            loaded = load_bundle(
                root / "models" / run_id / "candidates" / reference.candidate_id,
                repository_root=root,
                expected_run_id=run_id,
                expected_candidate_id=reference.candidate_id,
                expected_model_sha256=reference.model_sha256,
                expected_metadata_sha256=reference.metadata_sha256,
                monitor=outer,
            )
            digest = hashlib.sha256()
            inference = MemoryMonitor(config.resources.memory_budget_mib)

            def validate(
                stream: BinaryIO,
                successful: MemoryMonitor = inference,
                expected: Callable[[], bytes] = digest.digest,
            ) -> None:
                successful.ensure_success()
                actual = hashlib.sha256()
                while chunk := stream.read(config.resources.read_buffer_bytes):
                    actual.update(chunk)
                if actual.digest() != expected():
                    raise ValidationError("Prediction publication changed")

            try:
                with atomic_writer(
                    directory / f"{role}.jsonl", root, validate=validate
                ) as stream:
                    with inference:

                        def predictions(
                            model: Any = loaded.model,
                            active: MemoryMonitor = inference,
                            ref: ModelReference = reference,
                            update: Callable[[bytes], None] = digest.update,
                        ) -> Iterator[SentencePrediction]:
                            for record in records:
                                confidence = score_sentence(
                                    record.feature_forms,
                                    model=model,
                                    config=config,
                                    monitor=active,
                                )
                                prediction = Prediction(
                                    1,
                                    run_id,
                                    ref,
                                    record.sentence_id,
                                    record.source_path,
                                    record.source_sha256,
                                    "test_gold",
                                    record.sentence_ordinal,
                                    record.token_ids,
                                    record.original_forms,
                                    record.normalized_forms,
                                    record.token_spans,
                                    confidence.labels,
                                    confidence.token_confidences,
                                )
                                encoded = jsonl_bytes(prediction.to_dict())
                                stream.write(encoded)
                                update(encoded)
                                yield SentencePrediction(
                                    record.sentence_id,
                                    record.token_ids,
                                    record.normalized_forms,
                                    confidence.labels,
                                )

                        models[role] = score_test(
                            records,
                            predictions(),
                            config=config,
                            vocabulary=vocabulary,
                            model_ref=reference,
                            final_evaluation=True,
                        )
            finally:
                loaded.close()
        from crf.utils.serialization import json_bytes

        report = dict(
            schema_version=1,
            run_id=run_id,
            split="test_gold",
            target_inventory=list(config.target_inventory),
            models=models,
        )
        validate_metric_report(report, config, split="test_gold")
        _publish(directory / "metrics.json", json_bytes(report), root)
        if load_development(root, run_id) != experiment:
            raise ValidationError("Frozen history changed during evaluation")
    outer.ensure_success()
    core = dict(
        schema_version=1,
        run_id=run_id,
        status="predictions_complete",
        started_at=context.started_at,
        finished_at=_utc_now(),
        development_sha256=file_sha256(experiment.run_path),
        selected_sha256=file_sha256(root / "models" / run_id / "selected.json"),
        outputs={name: file_sha256(directory / name) for name in CORE_OUTPUTS},
        environment=dict(
            python_version=context.python_version,
            unicode_version=context.unicode_version,
            dependency_versions=dict(context.dependency_versions),
            source_revision=context.source_revision,
            source_dirty=context.source_dirty,
        ),
        peak_rss_mib=outer.peak_rss_mib,
    )
    _publish(directory / "core.json", json_bytes(core), root)
    return _load_core(experiment, root)


@dataclass(frozen=True)
class FinalEvaluation:
    core: EvaluationCore
    comparison: dict[str, object]
    completion_path: Path


def _bootstrap_inputs(
    core: EvaluationCore, monitor: MemoryMonitor
) -> tuple[Outcome, ...]:
    from crf.bootstrap import Outcome

    records = tuple(
        s.to_record(core.experiment.config, final_evaluation=True) for s in core.gold
    )
    gate = cached_predictions(
        core.directory / "gate.jsonl",
        records,
        core.experiment,
        core.experiment.selection.gate,
        monitor,
    )
    final = cached_predictions(
        core.directory / "final.jsonl",
        records,
        core.experiment,
        core.experiment.selection.final,
        monitor,
    )
    return tuple(
        Outcome(
            record.sentence_id,
            len(record.token_ids),
            sum(
                a == b
                for a, b in zip(record.target_labels, g.predicted_labels, strict=True)
            ),
            sum(
                a == b
                for a, b in zip(record.target_labels, f.predicted_labels, strict=True)
            ),
        )
        for record, g, f in zip(records, gate, final, strict=True)
    )


def _differences(core: EvaluationCore) -> tuple[float, float]:
    models = cast(dict[str, dict[str, float]], core.metrics["models"])
    return (
        100 * (models["final"]["accuracy"] - models["gate"]["accuracy"]),
        100 * (models["final"]["accuracy"] - models["gold_control"]["accuracy"]),
    )


def _validate_final(
    value: object,
    core: EvaluationCore,
    root: Path,
    monitor: MemoryMonitor | None = None,
) -> FinalEvaluation:
    import math

    from crf.bootstrap import resampling_units
    from crf.utils.serialization import integer, text_value

    raw = object_fields(
        value,
        {
            "schema_version",
            "run_id",
            "status",
            "started_at",
            "finished_at",
            "development_sha256",
            "selected_sha256",
            "outputs",
            "environment",
            "peak_rss_mib",
            "error",
        },
        "evaluation completion",
    )
    if (
        integer(raw["schema_version"], "schema_version") != 1
        or raw["run_id"] != core.experiment.run_id
        or raw["status"] not in ("complete", "failed")
    ):
        raise ValidationError("Invalid evaluation completion")
    validate_observation(
        raw["environment"], raw["started_at"], raw["finished_at"], raw["peak_rss_mib"]
    )
    if raw["development_sha256"] != file_sha256(
        core.experiment.run_path, monitor
    ) or raw["selected_sha256"] != file_sha256(
        root / "models" / core.experiment.run_id / "selected.json", monitor
    ):
        raise ValidationError("Evaluation development/selection links changed")
    if raw["status"] == "failed":
        error = object_fields(raw["error"], {"stage", "message"}, "error")
        text_value(error["stage"], "error stage")
        text_value(error["message"], "error message")
        raise ValidationError("Failed evaluation cannot authorize cached success")
    if raw["error"] is not None:
        raise ValidationError("Successful evaluation requires null error")
    outputs = object_fields(
        raw["outputs"],
        set(CORE_OUTPUTS) | {"comparison.json", "units.jsonl"},
        "evaluation outputs",
    )
    for name, checksum in outputs.items():
        if checksum != file_sha256(core.directory / name, monitor):
            raise ValidationError("Evaluation output checksum mismatch")
    unit_type, units, fallback = resampling_units(core.gold)
    if read_bytes(core.directory / "units.jsonl") != b"".join(
        jsonl_bytes(unit.to_dict()) for unit in units
    ):
        raise ValidationError(
            "Persisted units disagree with gold document/sentence grouping"
        )
    comparison = object_fields(
        loads_json(read_bytes(core.directory / "comparison.json")),
        {
            "schema_version",
            "run_id",
            "split",
            "final_minus_gate_pp",
            "final_minus_gold_control_pp",
            "bootstrap",
        },
        "comparison",
    )
    if (
        integer(comparison["schema_version"], "schema_version") != 1
        or comparison["run_id"] != core.experiment.run_id
        or comparison["split"] != "test_gold"
    ):
        raise ValidationError("Invalid comparison identity")
    for key, expected in zip(
        ("final_minus_gate_pp", "final_minus_gold_control_pp"),
        _differences(core),
        strict=True,
    ):
        actual = comparison[key]
        if type(actual) not in (int, float) or actual != expected:
            raise ValidationError("Signed comparison disagrees with metrics")
    report = object_fields(
        comparison["bootstrap"],
        {
            "unit_type",
            "unit_count",
            "resamples",
            "seed",
            "method",
            "percentile_interpolation",
            "lower_pp",
            "upper_pp",
            "unavailable_reason",
        },
        "bootstrap",
    )
    config = core.experiment.config.evaluation
    if (
        report["unit_type"] != unit_type
        or integer(report["unit_count"], "unit count") != len(units)
        or integer(report["resamples"], "resamples", 1) != config.bootstrap_resamples
        or integer(report["seed"], "seed") != config.bootstrap_seed
        or report["method"] != "percentile"
        or report["percentile_interpolation"] != "linear"
    ):
        raise ValidationError("Bootstrap frozen settings mismatch")
    if len(units) < 2:
        if report["lower_pp"] is not None or report["upper_pp"] is not None:
            raise ValidationError("Insufficient units require unavailable interval")
        text_value(report["unavailable_reason"], "unavailable reason")
    else:
        bounds = (report["lower_pp"], report["upper_pp"])
        if (
            any(
                type(v) not in (int, float)
                or not math.isfinite(cast(float, v))
                or not -100 <= cast(float, v) <= 100
                for v in bounds
            )
            or cast(float, bounds[0]) > cast(float, bounds[1])
            or report["unavailable_reason"] is not None
        ):
            raise ValidationError("Invalid bootstrap interval")
    diagnostic = loads_json(read_bytes(core.directory / "diagnostics.json"))
    if diagnostic != dict(
        schema_version=1,
        run_id=core.experiment.run_id,
        unit_type=unit_type,
        unit_count=len(units),
        fallback_reason=fallback,
    ):
        raise ValidationError("Bootstrap fallback diagnostics disagree")
    return FinalEvaluation(core, comparison, core.directory / "completion.json")


def evaluate(root: Path, run_id: str, *, seed: int | None = None) -> FinalEvaluation:
    from crf.bootstrap import paired_bootstrap, resampling_units
    from crf.reporting import validate_failure
    from crf.utils.artifacts import load_completed, publish_completion
    from crf.utils.serialization import json_bytes

    root = root.resolve()
    context = _context(root)
    experiment = load_development(root, run_id)
    monitor = MemoryMonitor(experiment.config.resources.memory_budget_mib)
    core: EvaluationCore | None = None
    cached: FinalEvaluation | None = None
    attempted = False

    def descriptor(error: BaseException | None = None) -> dict[str, object]:
        assert core is not None
        return dict(
            schema_version=1,
            run_id=run_id,
            status="complete" if error is None else "failed",
            started_at=context.started_at,
            finished_at=_utc_now(),
            development_sha256=file_sha256(experiment.run_path),
            selected_sha256=file_sha256(root / "models" / run_id / "selected.json"),
            outputs={}
            if error is not None
            else {
                name: file_sha256(core.directory / name)
                for name in (*CORE_OUTPUTS, "comparison.json", "units.jsonl")
            },
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
            else dict(stage="evaluation", message=str(error) or type(error).__name__),
        )

    try:
        with monitor:
            core = evaluate_core(root, run_id, seed=seed)
            path = core.directory / "completion.json"
            if path.exists() or path.is_symlink():
                cached = load_completed(
                    path,
                    root,
                    validate=lambda value: _validate_final(value, core, root, monitor),
                )
            else:
                marker = core.directory / "bootstrap_attempt.json"
                if marker.exists() or marker.is_symlink():
                    raise ValidationError(
                        "Interrupted bootstrap attempt cannot be resumed automatically"
                    )
                _publish(
                    marker,
                    json_bytes(
                        dict(
                            schema_version=1,
                            run_id=run_id,
                            started_at=context.started_at,
                        )
                    ),
                    root,
                )
                attempted = True
                unit_type, units, fallback = resampling_units(core.gold)
                outcomes = _bootstrap_inputs(core, monitor)
                config = experiment.config.evaluation
                report = paired_bootstrap(
                    units,
                    outcomes,
                    unit_type=unit_type,
                    resamples=config.bootstrap_resamples,
                    seed=config.bootstrap_seed,
                    monitor=monitor,
                )
                differences = _differences(core)
                comparison = dict(
                    schema_version=1,
                    run_id=run_id,
                    split="test_gold",
                    final_minus_gate_pp=differences[0],
                    final_minus_gold_control_pp=differences[1],
                    bootstrap=report,
                )
                _publish(
                    core.directory / "units.jsonl",
                    b"".join(jsonl_bytes(unit.to_dict()) for unit in units),
                    root,
                )
                _publish(
                    core.directory / "comparison.json", json_bytes(comparison), root
                )
                _publish(
                    core.directory / "diagnostics.json",
                    json_bytes(
                        dict(
                            schema_version=1,
                            run_id=run_id,
                            unit_type=unit_type,
                            unit_count=len(units),
                            fallback_reason=fallback,
                        )
                    ),
                    root,
                )
                if load_development(root, run_id) != experiment:
                    raise ValidationError("Frozen development changed during bootstrap")
        monitor.ensure_success()
    except BaseException as exc:
        if core is not None and attempted:
            publish_completion(
                core.directory / "completion.json",
                descriptor(exc),
                root,
                validate=validate_failure,
                monitor=monitor,
            )
        raise
    if cached is not None:
        return cached
    assert core is not None
    path = core.directory / "completion.json"
    completed = descriptor()
    publish_completion(
        path,
        completed,
        root,
        validate=lambda value: _validate_final(value, core, root),
        monitor=monitor,
    )
    return _validate_final(completed, core, root)
