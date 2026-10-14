"""Load saved transforms and document the experiment artifact directory tree."""

import json
from pathlib import Path
import sys
from threading import RLock

import joblib

from . import vectors


_LOAD_LOCK = RLock()


def load_transform(path):
    """Load a trusted joblib transform, including pre-package encoders.

    Older files reference ``vectors.FeatureEncoder``. Temporarily expose the
    relocated module during unpickling so those bytes need not be rewritten.
    New files reference ``postag.vectors.FeatureEncoder`` directly.
    """
    with _LOAD_LOCK:
        added_alias = "vectors" not in sys.modules
        if added_alias:
            sys.modules["vectors"] = vectors
        try:
            return joblib.load(path)
        finally:
            if added_alias:
                del sys.modules["vectors"]


def write_run_readmes(output):
    """Add a README at each artifact level without replacing existing docs."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    old_layout = "main.py" in manifest.get("code_sha256", {})
    contents = {
        output: "# Experiment run\n\n"
        "Start with [report.md](report.md) for the comparison and cluster links.\n\n"
        "- `manifest.json`: parameters, input/code hashes, dependencies, and timings.\n"
        "- `rows.jsonl`: original token occurrences in shared matrix row order.\n"
        "- `features.jsonl.gz`: feature dictionaries in that same order.\n"
        "- `word_frequencies.json`: full-corpus TF table.\n"
        "- `results.json`: cluster statistics and reference metrics.\n"
        "- `dictionary.json` / `reference_labels.jsonl`: dictionary provenance and labels, when used.\n"
        "- `common_non_noise_row_ids.json`: common scored rows, when reference evaluation is available.\n\n"
        "Feature variants are separate child directories. Keep each run intact: row order\n"
        "connects matrices and assignments. Use a new output directory for a new run.\n\n"
        "Load trusted fitted transforms with `postag.artifacts.load_transform`; it\n"
        "supports both current and pre-package artifacts.\n"
    }
    if old_layout:
        contents[output] += (
            "\nThis run predates the directory reorganization. Its original artifacts and\n"
            "manifest are preserved. Historical input paths ending in `tokenized/` now\n"
            "refer to `data/tokenized/`; source code moved into `postag/`, with `main.py`\n"
            "renamed to `features.py`. Recorded hashes describe the original run, not\n"
            "the reorganized source. The legacy encoder loader handles the old module name.\n"
        )
    for variant in sorted(p for p in output.iterdir() if p.is_dir() and not p.is_symlink()):
        contents[variant] = (
            f"# {variant.name} feature vectors\n\n"
            "[Run overview](../README.md)\n\n"
            "- `features.npz`: sparse numerical matrix (load with `scipy.sparse.load_npz`).\n"
            "- `feature_names.json`: matrix column names.\n"
            "- `encoder.joblib`: fitted feature selector, vocabulary, and numeric scales;\n"
            "  load with `postag.artifacts.load_transform`.\n\n"
            "Rows align with [rows.jsonl](../rows.jsonl). Each `svd-*` child holds one\n"
            "dimensionality-reduction experiment. Feature definitions are in the run manifest.\n"
        )
        for reduction in sorted(p for p in variant.iterdir() if p.is_dir() and not p.is_symlink()):
            contents[reduction] = (
                f"# {variant.name} / {reduction.name}\n\n"
                "[Feature matrix](../README.md)\n\n"
                "- `vectors.npy`: dense reduced vectors; load with `numpy.load(..., allow_pickle=False)`.\n"
                "- `reducer.joblib`: fitted SVD and optional normalization transform.\n\n"
                "The directory name gives the requested dimensions; the manifest records\n"
                "the actual dimensions when small inputs require a cap. Rows align with\n"
                "[rows.jsonl](../../rows.jsonl). Each `mcs-*-ms-*` child contains one\n"
                "HDBSCAN configuration applied to these vectors.\n"
            )
            for clustering in sorted(p for p in reduction.iterdir() if p.is_dir() and not p.is_symlink()):
                contents[clustering] = (
                    f"# HDBSCAN: {clustering.name}\n\n"
                    "[Input vectors](../README.md)\n\n"
                    "The name encodes minimum cluster size (`mcs`) and minimum samples (`ms`).\n\n"
                    "- [inspection.md](inspection.md): representative forms and sentence windows.\n"
                    "- [clusters.json](clusters.json): sizes, POS distributions, suffix counts, and examples.\n"
                    "- `assignments.npz`: labels, membership strengths, and explicit row IDs;\n"
                    "  load with `numpy.load(..., allow_pickle=False)`.\n\n"
                    "Noise has label `-1`; membership strength is not a POS probability.\n"
                    "Scores are in the run-level [results.json](../../../results.json).\n"
                )
    for directory, content in contents.items():
        path = directory / "README.md"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
