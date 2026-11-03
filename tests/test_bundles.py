"""Offline persistence fixtures using locked CRFsuite/joblib; no production data."""

import copy
import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import pytest

from crf.data.identity import InputIdentity, sentence_id
from crf.data.records import SentenceRecord, parse_sentence_record
from crf.features.surface import sentence_features
from crf.models import bundles
from crf.models.bundles import BundleMetadata, load_bundle, parse_metadata, save_bundle
from crf.models.training import FittedCandidate, fit_candidate
from crf.utils.config import PipelineConfig, parse_config
from crf.utils.resources import (
    MemoryBudgetExceeded,
    MemoryMonitor,
    ResourceMonitoringError,
)
from crf.utils.serialization import ValidationError, json_bytes, loads_json


def train(config: PipelineConfig) -> tuple[FittedCandidate, MemoryMonitor]:
    source = InputIdentity(
        1, "train_gold", config.data.train_gold.as_posix(), "a" * 64, 100, "utf-8"
    )
    labels = ("ADP", "NOUN", "VERB")
    value = SentenceRecord(
        1,
        "gold",
        sentence_id(source.source_role, source.source_path, source.source_sha256, 1),
        source.source_path,
        source.source_sha256,
        "train_gold",
        1,
        None,
        "У Києві живе",
        "original",
        (1, 2, 3),
        ("У", "Києві", "живе"),
        ("У", "Києві", "живе"),
        (None,) * 3,
        labels,
        config.label_space.map_upos(labels),
        None,
    ).to_dict()
    record = parse_sentence_record(value, config=config, source=source)
    with MemoryMonitor(
        config.resources.memory_budget_mib, sampler=lambda: 1, interval_seconds=60
    ) as monitor:
        candidate = fit_candidate(
            [record], config=config, c1=0.1, c2=0.1, monitor=monitor
        )
    monitor.ensure_success()
    return candidate, monitor


@pytest.fixture(autouse=True)
def controlled_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bundles,
        "MemoryMonitor",
        lambda budget: MemoryMonitor(budget, sampler=lambda: 1, interval_seconds=60),
    )


@pytest.fixture
def candidate(artifact_config: PipelineConfig) -> tuple[FittedCandidate, MemoryMonitor]:
    return train(artifact_config)


def save(
    root: Path,
    config: PipelineConfig,
    candidate: tuple[FittedCandidate, MemoryMonitor],
    **kwargs: Any,
) -> BundleMetadata:
    fitted, monitor = candidate
    return save_bundle(
        fitted,
        config=config,
        repository_root=root,
        run_id="run",
        candidate_id="candidate",
        monitor=monitor,
        **kwargs,
    )


@pytest.fixture
def saved(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    candidate: tuple[FittedCandidate, MemoryMonitor],
) -> tuple[Path, BundleMetadata]:
    metadata = save(artifact_repository, artifact_config, candidate)
    return artifact_repository / "models/run/candidates/candidate", metadata


def sidecar(directory: Path) -> dict[str, Any]:
    return loads_json((directory / "metadata.json").read_bytes())


def rewrite(directory: Path, value: dict[str, Any]) -> None:
    # Deliberate fixture corruption; never operate on real published artifacts.
    (directory / "metadata.json").write_bytes(json_bytes(value))


@pytest.mark.integration
def test_native_bundle_round_trip(
    saved: tuple[Path, BundleMetadata],
    candidate: tuple[FittedCandidate, MemoryMonitor],
    artifact_repository: Path,
) -> None:
    directory, metadata = saved
    metadata_hash = hashlib.sha256(
        (directory / "metadata.json").read_bytes()
    ).hexdigest()
    assert (
        hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
        == metadata.model_sha256
    )
    assert sidecar(directory) == metadata.to_dict()
    assert parse_metadata(metadata.to_dict()) == metadata
    loaded = load_bundle(
        directory,
        repository_root=artifact_repository,
        expected_run_id="run",
        expected_candidate_id="candidate",
        expected_model_sha256=metadata.model_sha256,
        expected_metadata_sha256=metadata_hash,
    )
    native = Path(loaded.model.modelfile.name)
    try:
        assert loaded.metadata == metadata
        assert loaded.metadata_sha256 == metadata_hash
        assert loaded.model.postag_training_record_ == candidate[0].record
        for forms in (["У", "Львові", "живе", "."], ["Нове’слово"], ["123", "!"]):
            features = sentence_features(forms)
            predicted = candidate[0].model.predict_single(features)
            assert loaded.model.predict_single(features) == predicted
            before = candidate[0].model.predict_marginals_single(features)
            after = loaded.model.predict_marginals_single(features)
            assert after == before
            assert [
                row[label] for row, label in zip(after, predicted, strict=True)
            ] == [row[label] for row, label in zip(before, predicted, strict=True)]
    finally:
        loaded.close()
    assert not native.exists()
    assert (directory / "model.joblib").is_file()


@pytest.mark.integration
def test_portable_bundle_without_original_sources(
    saved: tuple[Path, BundleMetadata],
    artifact_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, metadata = saved
    portable = tmp_path / "copied-bundle"
    shutil.copytree(directory, portable)
    shutil.rmtree(artifact_repository / "data")
    shutil.rmtree(artifact_repository / "models")
    original_open = Path.open

    def no_sources(self: Path, *args: Any, **kwargs: Any) -> Any:
        assert "data/raw" not in self.as_posix()
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_sources)
    loaded = load_bundle(portable)
    try:
        assert loaded.metadata == metadata
        assert loaded.model.predict_single(
            sentence_features(["У", "Києві", "живе"])
        ) == ["ADP", "NOUN", "VERB"]
    finally:
        loaded.close()


def test_metadata_has_no_source_filesystem_dependency(
    saved: tuple[Path, BundleMetadata], monkeypatch: pytest.MonkeyPatch
) -> None:
    value = saved[1].to_dict()
    for method in ("resolve", "stat", "lstat", "open"):
        original = getattr(Path, method)

        def guarded(
            self: Path, *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            assert "data/raw" not in self.as_posix(), (
                "Metadata inspected original paths"
            )
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method, guarded)
    assert parse_metadata(value) == saved[1]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["mapped", "iteration_limit", "unlimited"])
def test_mapped_and_stopping_variants(
    artifact_repository: Path, artifact_config: PipelineConfig, mode: str
) -> None:
    config = artifact_config
    if mode == "mapped":
        config = replace(
            config,
            tag_mapping=tuple(
                (tag, "FUNC") for tag in ("ADP", "CCONJ", "PART", "SCONJ")
            ),
        )
    else:
        config = replace(
            config,
            training=replace(
                config.training,
                parameters=replace(
                    config.training.parameters,
                    max_iterations=1 if mode == "iteration_limit" else 0,
                ),
            ),
        )
    fitted = train(config)
    metadata = save(artifact_repository, config, fitted)
    loaded = load_bundle(artifact_repository / "models/run/candidates/candidate")
    try:
        assert loaded.metadata == metadata
        if mode == "mapped":
            assert (
                "FUNC" in loaded.model.classes_ and "ADP" not in loaded.model.classes_
            )
            assert len(metadata.training.target_inventory) == 14
        elif mode == "iteration_limit":
            assert metadata.training.termination_reason == "iteration_limit"
            assert metadata.training.iterations == 1
        else:
            assert metadata.training.termination_reason == "converged"
    finally:
        loaded.close()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"run_id": "../run"},
        {"candidate_id": "київ"},
        {"model_sha256": "A" * 64},
        {"training_dataset_sha256": "bad"},
        {"training_sentence_count": True},
        {"training_sentence_count": 0},
        {"training_sentence_count": 4},
        {"training_token_count": 0},
        {"pseudo_sentence_count": -1},
        {"pseudo_sentence_count": 1},
        {"pseudo_token_count": 1},
        {
            "training_sentence_count": 2,
            "pseudo_sentence_count": 1,
            "pseudo_token_count": 3,
        },
        {"iterations": True},
        {"iterations": -1},
        {"iterations": 201},
        {"termination_reason": "failed"},
        {"termination_reason": "iteration_limit"},
        {"seed": True},
        {"seed": 1},
        {"python_version": "wrong"},
        {"unicode_version": "wrong"},
        {"dependency_versions": {}},
        {"dependency_versions": {"joblib": "wrong"}},
        {"target_inventory": ["NOUN"]},
        {"crf_parameters": {}},
        {"effective_config": {}},
        {"unexpected": None},
    ],
)
def test_bad_sidecars_rejected_before_deserialization(
    saved: tuple[Path, BundleMetadata],
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
) -> None:
    directory, _ = saved
    value = sidecar(directory)
    value.update(change)
    rewrite(directory, value)
    monkeypatch.setattr(
        bundles.joblib,
        "load",
        lambda *args, **kwargs: pytest.fail("Bad metadata reached joblib"),
    )
    with pytest.raises(ValidationError):
        load_bundle(directory)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("algorithm", "l2sgd"),
        ("c1", True),
        ("c1", 0.2),
        ("c2", "0.1"),
        ("max_iterations", True),
        ("max_iterations", 2),
        ("epsilon", 1.0),
        ("trainer_cls", "callback"),
        ("model_filename", "original.crfsuite"),
        ("extra", None),
    ],
)
def test_bad_estimator_parameters(
    saved: tuple[Path, BundleMetadata], field: str, bad: object
) -> None:
    directory, _ = saved
    value = sidecar(directory)
    value["crf_parameters"][field] = bad
    rewrite(directory, value)
    with pytest.raises(ValidationError):
        load_bundle(directory)


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("features", "schema_version", 2),
        ("preprocessing", "normalizer_version", 2),
        ("preprocessing", "tokenizer_adapter_version", 2),
        ("data", "train_gold", "../train.conllu"),
        ("data", "train_gold", "/data/raw/gold/train.conllu"),
        ("data", "train_gold", "data/raw/gold/./train.conllu"),
        ("data", "source_encoding", "bad-codec"),
    ],
)
def test_preprocessing_and_recorded_paths(
    saved: tuple[Path, BundleMetadata], section: str, key: str, bad: object
) -> None:
    directory, _ = saved
    value = sidecar(directory)
    value["effective_config"][section][key] = bad
    rewrite(directory, value)
    with pytest.raises(ValidationError):
        load_bundle(directory)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_field",
        "partial_config",
        "duplicates",
        "nested_duplicates",
        "nan",
        "bom",
        "invalid_utf8",
        "truncated",
    ],
)
def test_strict_json_and_complete_settings(
    saved: tuple[Path, BundleMetadata], corruption: str
) -> None:
    directory, _ = saved
    value = sidecar(directory)
    content = (directory / "metadata.json").read_bytes()
    if corruption == "missing_field":
        del value["seed"]
        content = json_bytes(value)
    elif corruption == "partial_config":
        del value["effective_config"]["features"]
        content = json_bytes(value)
    elif corruption == "duplicates":
        content = b'{"seed":42,' + content[1:]
    elif corruption == "nested_duplicates":
        content = content.replace(b'"c1":0.1', b'"c1":0.1,"c1":0.1')
    elif corruption == "nan":
        content = content.replace(b'"c1":0.1', b'"c1":NaN')
    elif corruption == "bom":
        content = b"\xef\xbb\xbf" + content
    elif corruption == "invalid_utf8":
        content += b"\xff"
    else:
        content = content[:20]
    (directory / "metadata.json").write_bytes(content)
    with pytest.raises(ValidationError):
        load_bundle(directory)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("expected_run_id", "wrong"),
        ("expected_candidate_id", "wrong"),
        ("expected_model_sha256", "b" * 64),
        ("expected_metadata_sha256", "b" * 64),
    ],
)
def test_frozen_reference_mismatches(
    saved: tuple[Path, BundleMetadata], field: str, bad: str
) -> None:
    with pytest.raises(ValidationError):
        load_bundle(saved[0], **{field: bad})


def test_wrong_bundle_location(
    saved: tuple[Path, BundleMetadata], tmp_path: Path
) -> None:
    copied = tmp_path / "models/run/candidates/another"
    shutil.copytree(saved[0], copied)
    with pytest.raises(ValidationError, match="location"):
        load_bundle(copied, repository_root=tmp_path)


@pytest.mark.parametrize(
    "mode",
    [
        "missing_model",
        "missing_metadata",
        "corrupt_bytes",
        "wrong_object",
        "invalid_pickle",
        "evidence_mismatch",
    ],
)
def test_incomplete_and_mismatched_models(
    saved: tuple[Path, BundleMetadata], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    directory, _ = saved
    value = sidecar(directory)
    model_path = directory / "model.joblib"
    if mode == "missing_model":
        model_path.unlink()
    elif mode == "missing_metadata":
        (directory / "metadata.json").unlink()
    elif mode == "corrupt_bytes":
        model_path.write_bytes(b"broken")
        monkeypatch.setattr(
            bundles.joblib,
            "load",
            lambda *args, **kwargs: pytest.fail("Checksum was not checked first"),
        )
    elif mode in ("wrong_object", "invalid_pickle"):
        if mode == "wrong_object":
            joblib.dump({"not": "a model"}, model_path)
        else:
            model_path.write_bytes(b"broken")
        value["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        rewrite(directory, value)
    else:
        value["training_dataset_sha256"] = "b" * 64
        rewrite(directory, value)
    with pytest.raises(ValidationError):
        load_bundle(directory)


@pytest.mark.parametrize("entry", ["model.joblib", "metadata.json", "directory"])
def test_symlinks_rejected(
    saved: tuple[Path, BundleMetadata], tmp_path: Path, entry: str
) -> None:
    directory, _ = saved
    if entry == "directory":
        link = tmp_path / "link"
        link.symlink_to(directory, target_is_directory=True)
        directory = link
    else:
        original = directory / entry
        target = tmp_path / entry
        original.rename(target)
        original.symlink_to(target)
    with pytest.raises(ValidationError):
        load_bundle(directory)


@pytest.mark.parametrize("name", ["model.joblib", "metadata.json"])
def test_special_files_rejected_without_blocking(
    saved: tuple[Path, BundleMetadata], name: str
) -> None:
    path = saved[0] / name
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValidationError, match="regular file"):
        load_bundle(saved[0])


def test_existing_candidate_preserved(
    saved: tuple[Path, BundleMetadata],
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    candidate: tuple[FittedCandidate, MemoryMonitor],
) -> None:
    directory, _ = saved
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError, match="already exists"):
        save(artifact_repository, artifact_config, candidate)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize(
    "mode", ["config", "counts", "model_options", "training_active", "training_failed"]
)
def test_save_rejects_inconsistent_training(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    candidate: tuple[FittedCandidate, MemoryMonitor],
    mode: str,
) -> None:
    fitted, monitor = candidate
    config = artifact_config
    if mode == "config":
        config = replace(config, seed=1)
    elif mode == "counts":
        fitted = replace(fitted, record=replace(fitted.record, training_token_count=9))
    elif mode == "model_options":
        fitted.model.c1 = 0.01
    elif mode == "training_active":
        monitor = MemoryMonitor(8192, sampler=lambda: 1)
    else:
        with (
            pytest.raises(RuntimeError),
            MemoryMonitor(8192, sampler=lambda: 1) as monitor,
        ):
            raise RuntimeError("failed training")
    with pytest.raises((ValueError, ResourceMonitoringError)):
        save(artifact_repository, config, (fitted, monitor))
    assert not (artifact_repository / "models").exists()


@pytest.mark.parametrize(
    "mode",
    ["dump", "round_trip", "serialization_memory", "serialization_exit", "sidecar"],
)
def test_failed_save_never_yields_loadable_bundle(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    candidate: tuple[FittedCandidate, MemoryMonitor],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    if mode == "dump":

        def broken(model: Any, stream: Any, **kwargs: Any) -> None:
            stream.write(b"partial")
            raise OSError("serialize failed")

        monkeypatch.setattr(bundles.joblib, "dump", broken)
    elif mode == "round_trip":
        original = bundles._signature
        calls = 0

        def signature(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            return result if calls == 1 else []

        monkeypatch.setattr(bundles, "_signature", signature)
    elif mode == "serialization_memory":
        monkeypatch.setattr(
            bundles,
            "MemoryMonitor",
            lambda budget: MemoryMonitor(budget, sampler=lambda: budget * 1024**2 + 1),
        )
    elif mode == "serialization_exit":

        class ExitFailure(MemoryMonitor):
            def __exit__(self, *args: Any) -> Any:
                super().__exit__(*args)
                raise MemoryBudgetExceeded("exit failure")

        monkeypatch.setattr(
            bundles,
            "MemoryMonitor",
            lambda budget: ExitFailure(budget, sampler=lambda: 1),
        )
    else:
        monkeypatch.setattr(
            bundles,
            "publish_bytes",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("sidecar failed")),
        )
    with pytest.raises((OSError, ValueError, ResourceMonitoringError)):
        save(artifact_repository, artifact_config, candidate)
    directory = artifact_repository / "models/run/candidates/candidate"
    assert not (directory / "metadata.json").exists()
    assert not list(directory.glob(".partial-*"))
    with pytest.raises(ValidationError):
        load_bundle(directory)
    if mode != "sidecar":
        assert not (directory / "model.joblib").exists()


def test_loader_uses_verified_snapshot(
    saved: tuple[Path, BundleMetadata], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = saved
    original = bundles.joblib.load

    def load(stream: Any) -> Any:
        # A later change to the source cannot replace bytes already verified.
        (directory / "model.joblib").write_bytes(b"changed after snapshot")
        return original(stream)

    monkeypatch.setattr(bundles.joblib, "load", load)
    loaded = load_bundle(directory)
    loaded.close()


def test_load_monitor_failure_cleans_native_temp(
    saved: tuple[Path, BundleMetadata], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = bundles._deserialize
    loaded_models = []
    failed = False

    def deserialize(*args: Any) -> Any:
        nonlocal failed
        result = original(*args)
        loaded_models.append(Path(result.modelfile.name))
        failed = True
        return result

    monkeypatch.setattr(bundles, "_deserialize", deserialize)
    with (
        pytest.raises(MemoryBudgetExceeded),
        MemoryMonitor(
            8192,
            sampler=lambda: 8192 * 1024**2 + 1 if failed else 1,
            interval_seconds=60,
        ) as monitor,
    ):
        load_bundle(saved[0], monitor=monitor)
    assert loaded_models and not loaded_models[0].exists()


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.conllu",
        "data/raw/gold/../train.conllu",
        "data/raw/gold//train.conllu",
        "data/raw/gold/./train.conllu",
        "data\\raw\\gold\\train.conllu",
    ],
)
def test_lexical_config_rejects_noncanonical_paths(
    artifact_config: PipelineConfig, path: str
) -> None:
    value = copy.deepcopy(artifact_config.to_dict())
    value["data"]["train_gold"] = path
    with pytest.raises(ValueError):
        parse_config(value, repository_root=Path(), resolve_paths=False)


@pytest.mark.parametrize(
    "mode", ["symlink_parent", "incomplete_directory", "invalid_id"]
)
def test_save_path_guards(
    artifact_repository: Path,
    artifact_config: PipelineConfig,
    candidate: tuple[FittedCandidate, MemoryMonitor],
    mode: str,
) -> None:
    if mode == "symlink_parent":
        (artifact_repository / "models").symlink_to(
            artifact_repository / "data/raw", target_is_directory=True
        )
    elif mode == "incomplete_directory":
        directory = artifact_repository / "models/run/candidates/candidate"
        directory.mkdir(parents=True)
        (directory / "model.joblib").write_bytes(b"preserve incomplete output")
    if mode == "invalid_id":
        with pytest.raises(ValueError):
            save_bundle(
                candidate[0],
                config=artifact_config,
                repository_root=artifact_repository,
                run_id="../escape",
                candidate_id="candidate",
                monitor=candidate[1],
            )
    else:
        with pytest.raises((OSError, ValueError)):
            save(artifact_repository, artifact_config, candidate)
    assert not (artifact_repository / "data/raw/run").exists()
    if mode == "incomplete_directory":
        assert (
            directory / "model.joblib"
        ).read_bytes() == b"preserve incomplete output"


def test_owned_loading_monitor_exit_failure_disposes_model(
    saved: tuple[Path, BundleMetadata],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    original = bundles._deserialize

    def deserialize(*args: Any) -> Any:
        result = original(*args)
        paths.append(Path(result.modelfile.name))
        return result

    class ExitFailure(MemoryMonitor):
        def __exit__(self, *args: Any) -> Any:
            super().__exit__(*args)
            raise MemoryBudgetExceeded("load exit failure")

    monkeypatch.setattr(bundles, "_deserialize", deserialize)
    monkeypatch.setattr(
        bundles, "MemoryMonitor", lambda budget: ExitFailure(budget, sampler=lambda: 1)
    )
    with pytest.raises(MemoryBudgetExceeded):
        load_bundle(saved[0])
    assert paths and not paths[0].exists()
