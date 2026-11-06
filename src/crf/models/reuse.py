"""Verified empty-augmentation reuse; no training or model inference."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from crf.evaluation import ModelReference, validate_metric_report
from crf.models.bundles import _open_file, parse_metadata
from crf.models.sampling import FrozenDataset
from crf.models.search import BaselineSearch, BaselineTrial, select_gate
from crf.utils.artifacts import publish_bytes
from crf.utils.config import PipelineConfig, parse_config
from crf.utils.resources import MemoryMonitor
from crf.utils.serialization import ValidationError, json_bytes, loads_json


@dataclass(frozen=True)
class ReusedTrial:
    trial_id: str
    threshold: float
    baseline: BaselineTrial

    def to_dict(self) -> dict[str, object]:
        result = self.baseline.to_dict()
        result.update(
            trial_id=self.trial_id,
            stage="final",
            threshold=self.threshold,
            status="reused",
        )
        return result


@dataclass(frozen=True)
class EmptyReuse:
    trials: tuple[ReusedTrial, ...]
    all_empty: bool
    retained_gate: ModelReference | None
    report_path: Path


def reuse_empty_selections(
    datasets: tuple[FrozenDataset, ...],
    baseline: BaselineSearch,
    *,
    config: PipelineConfig,
    repository_root: Path,
    run_id: str,
) -> EmptyReuse:
    """Validate cached evidence and record reuse without fitting or rescoring.

    Nonempty threshold fits and combined trial completion remain M7-owned.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValidationError("Invalid reuse run ID")
    if (
        parse_config(
            config.to_dict(), repository_root=repository_root, resolve_paths=False
        )
        != config
    ):
        raise ValidationError("Noncanonical reuse configuration")
    root = repository_root.resolve()
    gate = select_gate(baseline.trials, config)
    if gate != baseline.gate or any(t.run_id != run_id for t in baseline.trials):
        raise ValidationError("Baseline run/gate mismatch")
    thresholds = sorted(config.self_training.thresholds)
    if [d.threshold for d in datasets] != thresholds:
        raise ValidationError("Incomplete or unordered threshold datasets")
    reused: list[ReusedTrial] = []
    with MemoryMonitor(config.resources.memory_budget_mib) as monitor:
        empty: list[tuple[int, FrozenDataset]] = []
        for index, dataset in enumerate(datasets):
            monitor.checkpoint()
            expected = root / "data/processed" / run_id / f"extended_{index:03d}.jsonl"
            if (
                dataset.path != expected
                or min(dataset.pseudo_sentences, dataset.pseudo_tokens) < 0
            ):
                raise ValidationError("Invalid frozen dataset reference")
            if bool(dataset.pseudo_sentences) != bool(dataset.pseudo_tokens):
                raise ValidationError("Inconsistent pseudo counts")
            if dataset.pseudo_sentences:
                continue
            digest = hashlib.sha256()
            tokens = sentences = 0
            with _open_file(dataset.path.parent, dataset.path.name) as stream:
                for line in stream:
                    monitor.checkpoint()
                    digest.update(line)
                    record = loads_json(line)
                    if (
                        not isinstance(record, dict)
                        or record.get("record_type") != "gold"
                        or record.get("source_role") != "train_gold"
                        or record.get("pseudo") is not None
                    ):
                        raise ValidationError(
                            "Empty selection must contain training gold only"
                        )
                    ids = record.get("token_ids")
                    if not isinstance(ids, list) or not ids:
                        raise ValidationError("Invalid gold token IDs")
                    tokens += len(ids)
                    sentences += 1
            if (
                digest.hexdigest() != dataset.sha256
                or tokens != dataset.gold_tokens
                or not sentences
            ):
                raise ValidationError("Frozen gold-only dataset mismatch")
            if any(
                t.training_dataset_sha256 != dataset.sha256 for t in baseline.trials
            ):
                raise ValidationError("Empty dataset differs from baseline training")
            empty.append((index, dataset))
        if empty:
            for trial in baseline.trials:
                monitor.checkpoint()
                reference = trial.model_ref
                assert reference is not None
                directory = (
                    root / "models" / run_id / "candidates" / reference.candidate_id
                )
                with _open_file(directory, "metadata.json") as stream:
                    encoded = stream.read()
                if hashlib.sha256(encoded).hexdigest() != reference.metadata_sha256:
                    raise ValidationError("Cached baseline metadata checksum mismatch")
                metadata = parse_metadata(loads_json(encoded))
                if (
                    metadata.run_id != run_id
                    or metadata.candidate_id != reference.candidate_id
                    or metadata.config != config
                    or metadata.model_sha256 != reference.model_sha256
                    or metadata.training.training_dataset_sha256
                    != trial.training_dataset_sha256
                    or any(
                        metadata.training.training_token_count != d.gold_tokens
                        for _, d in empty
                    )
                    or metadata.training.pseudo_token_count != 0
                    or metadata.training.pseudo_sentence_count != 0
                    or metadata.training.crf_parameters != trial.crf_parameters
                ):
                    raise ValidationError("Cached baseline training evidence mismatch")
                digest = hashlib.sha256()
                with _open_file(directory, "model.joblib") as stream:
                    while chunk := stream.read(config.resources.read_buffer_bytes):
                        monitor.checkpoint()
                        digest.update(chunk)
                if digest.hexdigest() != reference.model_sha256:
                    raise ValidationError("Cached baseline model checksum mismatch")
                report_dir = root / "reports" / run_id / "validation" / trial.trial_id
                with _open_file(report_dir, "metrics.json") as stream:
                    report = validate_metric_report(loads_json(stream.read()), config)
                models = cast(dict[str, dict[str, object]], report["models"])
                if report["run_id"] != run_id or set(models) != {trial.trial_id}:
                    raise ValidationError(
                        "Cached baseline validation identity mismatch"
                    )
                scores = models[trial.trial_id]
                if (
                    scores["model_ref"] != reference.to_dict()
                    or scores["accuracy"] != trial.validation_accuracy
                    or scores["macro_f1"] != trial.validation_macro_f1
                ):
                    raise ValidationError("Cached baseline validation scores mismatch")
            for index, dataset in empty:
                for candidate_index, trial in enumerate(baseline.trials):
                    reused.append(
                        ReusedTrial(
                            f"final_{index:03d}_{candidate_index:03d}",
                            dataset.threshold,
                            trial,
                        )
                    )
    monitor.ensure_success()
    all_empty = len(empty) == len(datasets)
    path = root / "reports" / run_id / "empty_reuse.json"
    content = json_bytes(
        dict(
            schema_version=1,
            run_id=run_id,
            self_training_status="not_performed_empty_selection"
            if all_empty
            else "nonempty_thresholds_pending",
            augmentation_gain_claimed=False,
            retained_gate=gate.to_dict() if all_empty else None,
            trials=[trial.to_dict() for trial in reused],
        )
    )

    def validate(stream: BinaryIO) -> None:
        monitor.ensure_success()
        if stream.read() != content:
            raise ValidationError("Reuse report changed")

    publish_bytes(path, content, root, validate=validate)
    return EmptyReuse(tuple(reused), all_empty, gate if all_empty else None, path)
