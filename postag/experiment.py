"""Run reproducible token-vector, HDBSCAN, and dictionary evaluation experiments."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from .artifacts import write_run_readmes
from .corpus import sample_rows, scan_corpus
from .evaluation import (cluster_details, evaluate_clusters, file_sha256,
                        load_dictionary, lookup_key, partition_scores, reference_labels)
from .features import PREFIX_LENGTHS, SUFFIX_LENGTHS, WORD_FEATURES
from .vectors import FeatureEncoder, VARIANTS, fit_reducer


PACKAGE_DIR = Path(__file__).resolve().parent
BASE = PACKAGE_DIR.parent
DEFAULT_DICTIONARY = BASE / "data/dict_uk/dict_corp_vis.txt.bz2"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_jsonl(path, records):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def log(message):
    print(message, file=sys.stderr, flush=True)


def format_score(value):
    return "n/a" if value is None else f"{value:.4f}"


def write_report(output, manifest, results):
    corpus = manifest["corpus"]
    lines = ["# Token vector experiment", "",
             f"Sample: **{manifest['sample']['rows']:,} token occurrences** from "
             f"{manifest['sample']['documents']} documents. Full corpus: "
             f"{corpus['tokens']:,} tokens in {len(corpus['documents'])} documents.", "",
             "Document-balanced sampling gives short documents more weight per token. "
             "TF uses the full corpus; vectorizer and SVD fit only the sampled rows. "
             "This is an exploratory comparison, not held-out POS accuracy.", "",
             "Original punctuation and capitalization are retained. The same rows are "
             "used for every configuration. HDBSCAN membership strengths are not POS probabilities.", "",
             "## Dictionary reference", ""]
    if manifest["dictionary"]["status"] == "ok":
        coverage = manifest["dictionary"]["coverage"]
        lines += ["Source and checksum: [dictionary.json](dictionary.json). "
                  "Labels use a documented coarse VESUM grouping, not UD labels.", "",
                  f"Lexical occurrences: {coverage['lexical_occurrences']:,}; "
                  f"dictionary coverage: {format_score(coverage['dictionary_coverage'])}; "
                  f"unambiguous coverage: {format_score(coverage['unambiguous_coverage'])}.", "",
                  "Occurrence counts: " + ", ".join(f"{k}={v:,}" for k, v in coverage["occurrences"].items()) + ".",
                  "Distinct normalized forms: " + ", ".join(
                      f"{k}={v:,}" for k, v in coverage["distinct_lookup_forms"].items()) + ".", "",
                  "Only unambiguous lexical occurrences are scored. Primary scores exclude noise. "
                  "Always compare assigned counts and noise coverage: a high score on a small "
                  "subset does not establish better overall features. Dictionary ambiguity "
                  "prevents treating these labels as contextual gold annotations.", ""]
    else:
        lines += ["**Pending:** " + manifest["dictionary"]["reason"], ""]
    lines += ["## Configuration comparison", "",
              "| Run | Clusters | Sample noise | Eligible assigned / total | POS noise | V-measure | ARI | Purity | Seconds |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        if result["status"] != "ok":
            lines.append(f"| {result['id']} | {result['status']}: {result['reason']} | | | | | | | |")
            continue
        evaluation = result.get("evaluation", {})
        scores = evaluation.get("non_noise", {})
        count = f"{evaluation.get('assigned_count', 'n/a')} / {evaluation.get('eligible_count', 'n/a')}"
        lines.append(f"| [{result['id']}]({result['id']}/clusters.json) | {result['clusters']} | "
                     f"{result['noise_fraction']:.1%} | {count} | "
                     f"{format_score(evaluation.get('eligible_noise_fraction'))} | "
                     f"{format_score(scores.get('v_measure'))} | {format_score(scores.get('ari'))} | "
                     f"{format_score(scores.get('purity'))} | {result['seconds']:.2f} |")
    eligible = [r for r in results if r.get("evaluation", {}).get("non_noise", {}).get("v_measure") is not None]
    if eligible:
        best = max(eligible, key=lambda r: r["evaluation"]["non_noise"]["v_measure"])
        scores = best["evaluation"]["non_noise"]
        lines += ["", f"Highest observed non-noise V-measure: **{best['id']}** "
                  f"({scores['v_measure']:.4f}, {scores['count']:,} scored occurrences). "
                  "This is an exploratory ranking with differing noise subsets, not a validated winner."]
        lines += ["", "`results.json` also includes noise-as-one-label scores, one-cluster baselines "
                  "on both the full eligible set and each assigned subset, and scores on the "
                  "intersection of non-noise eligible rows across successful runs."]
    numeric_warnings = sorted({warning for matrix in manifest["matrices"]
                               for reduction in matrix["reductions"] for warning in reduction["warnings"]})
    if numeric_warnings:
        lines += ["", "## Numerical diagnostics", "",
                  "SVD emitted the following warnings. Fitted parameters and vectors passed "
                  "finite-value checks and component orthogonality checks; details are in the manifest.", ""]
        lines += ["- " + warning for warning in numeric_warnings]
    lines += ["", "## Inspecting vectors and clusters", "",
              "- `rows.jsonl`: global row order, original tokens, source positions, and nearby context.",
              "- `features.jsonl.gz`: original feature dictionaries in the same row order.",
              "- `word_frequencies.json`: full-corpus lowercase TF lookup.",
              "- `<variant>/features.npz` and `encoder.joblib`: sparse matrix and fitted encoder.",
              "- `<variant>/svd-<requested>/vectors.npy` and `reducer.joblib`: reduced vectors and fitted transform.",
              "- Each run's `assignments.npz`: labels, membership strengths, and row IDs.",
              "- Each run's `clusters.json` and `inspection.md`: cluster sizes, distinct forms, "
              "POS distributions, ending counts, and representative contexts (including noise).", "",
              "See `manifest.json` for input/code hashes, dependency versions, parameters, "
              "numeric scales, explained variance, matrix storage, and timings. "
              "Dense-memory estimates are lower bounds, not estimates of HDBSCAN's total peak RAM.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_inspection(path, details):
    lines = ["# Cluster inspection", "",
             "Examples favor distinct forms with high membership strength. Context is an "
             "original token window (spacing is reconstructed); source positions are in clusters.json.", ""]
    for group in details:
        title = "Noise" if group["is_noise"] else f"Cluster {group['label']}"
        lines += [f"## {title}", "",
                  f"{group['size']} occurrences; {group['distinct_forms']} distinct forms.", "",
                  "Top forms: " + ", ".join(f"{word!r} ({count})" for word, count in group["top_forms"]),
                  "", "Unambiguous POS counts: " + json.dumps(group["pos_counts"], ensure_ascii=False),
                  "", "Ending counts: " + json.dumps(group["ending_counts"], ensure_ascii=False), ""]
        for example in group["examples"]:
            window = list(example["context_tokens"])
            position = example["token_index"] - example["context_start"]
            window[position] = "[" + window[position] + "]"
            # JSON quoting keeps source Markdown/line breaks as literal readable data.
            lines.append("- " + json.dumps(" ".join(window), ensure_ascii=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args):
    started = time.perf_counter()
    root = args.input.resolve()
    output = args.output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output and input directories must not contain each other")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new run directory: {output}")
    log("Scanning tokenized corpus and computing full-corpus TF...")
    frequencies, corpus = scan_corpus(root, args.rows)
    rows, features = sample_rows(root, corpus, frequencies, args.sample_size, args.seed)
    if len(rows) < 3:
        raise ValueError("At least three selected token occurrences are required")
    corpus_seconds = time.perf_counter() - started
    log(f"Selected {len(rows):,} / {corpus['tokens']:,} tokens across "
        f"{len({r['document'] for r in rows})} documents.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(root), "configuration": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
        "python": platform.python_version(),
        "versions": {name: version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "code_sha256": {f"postag/{name}": file_sha256(PACKAGE_DIR / name) for name in
                        ("features.py", "corpus.py", "vectors.py", "evaluation.py",
                         "experiment.py", "artifacts.py")},
        "feature_definitions": {"prefix_lengths": PREFIX_LENGTHS,
                                "suffix_lengths": SUFFIX_LENGTHS,
                                "word_features": list(WORD_FEATURES),
                                "bias": False, "sentence_boundaries": "BOS/EOS"},
        "corpus": corpus, "corpus_seconds": corpus_seconds,
        "sample": {"rows": len(rows), "documents": len({r["document"] for r in rows}),
                   "method": "balanced document quotas; seeded uniform occurrences within each document",
                   "seed": args.seed, "token_kinds": dict(Counter(r["kind"] for r in rows))},
        "hdbscan": {"implementation": "sklearn.cluster.HDBSCAN", "algorithm": "kd_tree",
                    "metric": "euclidean", "cluster_selection_method": "eom",
                    "min_samples_includes_self": True, "allow_single_cluster": False},
        "dictionary": {"status": "pending", "reason": "No dictionary supplied; use --dictionary PATH."},
        "matrices": [],
    }
    save_json(output / "manifest.json", manifest)
    results = []
    try:
        save_jsonl(output / "rows.jsonl", rows)
        save_jsonl(output / "features.jsonl.gz", features)
        save_json(output / "word_frequencies.json", frequencies)
        references = None
        if args.dictionary:
            log("Reading dictionary analyses for sampled lexical forms...")
            lookup, provenance = load_dictionary(
                args.dictionary, {lookup_key(r["token"]) for r in rows if r["kind"] == "lexical"},
                args.dictionary_source, args.dictionary_format)
            references, coverage = reference_labels(rows, lookup)
            save_json(output / "dictionary.json", provenance)
            save_jsonl(output / "reference_labels.jsonl", references)
            manifest["dictionary"] = {"status": "ok", "coverage": coverage}
            log(f"Dictionary: {coverage['occurrences']}.")

        all_labels = []
        for variant in args.variants:
            log(f"Encoding {variant} features...")
            encoded_at = time.perf_counter()
            directory = output / variant
            directory.mkdir()
            encoder = FeatureEncoder(variant, args.numeric_scaling)
            matrix = encoder.fit_transform(features)
            sparse.save_npz(directory / "features.npz", matrix)
            joblib.dump(encoder, directory / "encoder.joblib")
            save_json(directory / "feature_names.json", encoder.vectorizer_.get_feature_names_out().tolist())
            matrix_info = {"variant": variant, "shape": list(matrix.shape), "nnz": matrix.nnz,
                           "sparse_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
                           "numeric_ranges_before_scaling": encoder.numeric_ranges_,
                           "encoding_seconds": time.perf_counter() - encoded_at, "reductions": []}
            manifest["matrices"].append(matrix_info)
            for components in args.components:
                reduced_at = time.perf_counter()
                reduced_dir = directory / f"svd-{components}"
                reduced_dir.mkdir()
                reducer, reduced, reduction = fit_reducer(matrix, components, args.seed, args.normalize)
                if reduction["warnings"]:
                    log("SVD warnings recorded in manifest; finite/orthogonality checks passed: "
                        + "; ".join(reduction["warnings"]))
                reduction.update({"dense_bytes": reduced.nbytes,
                                  "full_eligible_dense_bytes_lower_bound":
                                  sum(d["eligible_tokens"] for d in corpus["documents"]) * reduced.shape[1] * 8,
                                  "seconds": time.perf_counter() - reduced_at})
                matrix_info["reductions"].append(reduction)
                joblib.dump(reducer, reduced_dir / "reducer.joblib")
                np.save(reduced_dir / "vectors.npy", reduced, allow_pickle=False)
                for cluster_size in args.min_cluster_sizes:
                    for min_samples in args.min_samples:
                        run_id = f"{variant}/svd-{components}/mcs-{cluster_size}-ms-{min_samples}"
                        result = {"id": run_id, "variant": variant, "components": reduction["components"],
                                  "min_cluster_size": cluster_size, "min_samples": min_samples}
                        if max(cluster_size, min_samples) > len(rows):
                            result.update(status="skipped", reason="HDBSCAN size exceeds sample size")
                            results.append(result)
                            continue
                        log(f"Clustering {run_id}...")
                        run_dir = output / run_id
                        run_dir.mkdir()
                        cluster_started = time.perf_counter()
                        clusterer = HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples,
                                            algorithm="kd_tree", metric="euclidean", n_jobs=1,
                                            allow_single_cluster=False, copy=True)
                        labels = clusterer.fit_predict(reduced)
                        probabilities = clusterer.probabilities_
                        seconds = time.perf_counter() - cluster_started
                        np.savez_compressed(run_dir / "assignments.npz", labels=labels,
                                            probabilities=probabilities,
                                            row_ids=np.array([r["row_id"] for r in rows]))
                        sizes = Counter(int(label) for label in labels if label >= 0)
                        result.update(status="ok", clusters=len(sizes),
                                      cluster_sizes=dict(sizes), noise_fraction=float(np.mean(labels < 0)),
                                      assigned_tokens=int(np.sum(labels >= 0)), seconds=seconds,
                                      outcome="all_noise" if not sizes else
                                      "single_cluster" if len(sizes) == 1 else "multiple_clusters")
                        if references is not None:
                            result["evaluation"] = evaluate_clusters(references, labels)
                        details = cluster_details(rows, labels, probabilities, references)
                        save_json(run_dir / "clusters.json", details)
                        write_inspection(run_dir / "inspection.md", details)
                        results.append(result)
                        all_labels.append((result, labels))
                        save_json(output / "results.json", results)
                save_json(output / "manifest.json", manifest)

        # A common assigned subset helps compare variants without changing the
        # scored population; it may be tiny/empty and must be reported as such.
        if references is not None and all_labels:
            common = np.array([r["status"] == "eligible" for r in references])
            for _, labels in all_labels:
                common &= labels >= 0
            truth = [r["pos"] for r, keep in zip(references, common) if keep]
            save_json(output / "common_non_noise_row_ids.json",
                      [r["row_id"] for r, keep in zip(rows, common) if keep])
            for result, labels in all_labels:
                result["evaluation"]["common_non_noise"] = partition_scores(truth, labels[common])
        manifest["status"] = "complete" if all_labels else "vectors_only"
        manifest["total_seconds"] = time.perf_counter() - started
        save_json(output / "results.json", results)
        save_json(output / "manifest.json", manifest)
        write_report(output, manifest, results)
        write_run_readmes(output)
        log(f"Finished: {output / 'report.md'}")
        return output
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        save_json(output / "manifest.json", manifest)
        raise


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "data" / "tokenized")
    parser.add_argument("--output", type=Path, default=BASE / "runs" /
                        datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Document-balanced token budget; 0 explicitly selects all tokens")
    parser.add_argument("--rows", choices=("all", "lexical"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["token", "context", "context_tf"])
    parser.add_argument("--components", nargs="+", type=positive_int, default=[30, 50])
    parser.add_argument("--min-cluster-sizes", nargs="+", type=positive_int, default=[15, 40])
    parser.add_argument("--min-samples", nargs="+", type=positive_int, default=[5, 10])
    parser.add_argument("--numeric-scaling", choices=("none", "std"), default="none")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize reduced vectors")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--dictionary", type=Path, help="Expanded dict_uk .txt or .txt.bz2 export")
    parser.add_argument("--dictionary-format", choices=("vis", "lt"), default="vis")
    parser.add_argument("--dictionary-source", type=Path, help="Provenance JSON containing SHA-256")
    parser.add_argument("--no-dictionary", action="store_true", help="Skip reference evaluation")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error("--sample-size must be zero or positive")
    if min(args.min_cluster_sizes) < 2:
        parser.error("--min-cluster-sizes must be at least 2")
    if args.no_dictionary and (args.dictionary or args.dictionary_source):
        parser.error("--no-dictionary cannot be combined with dictionary paths")
    if not args.no_dictionary and args.dictionary is None and DEFAULT_DICTIONARY.exists():
        args.dictionary = DEFAULT_DICTIONARY
    if args.dictionary and args.dictionary.resolve() == DEFAULT_DICTIONARY.resolve() and not args.dictionary_source:
        args.dictionary_source = DEFAULT_DICTIONARY.parent / "source.json"
    if args.dictionary_source and not args.dictionary:
        parser.error("--dictionary-source requires a dictionary")
    for key in ("variants", "components", "min_cluster_sizes", "min_samples"):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    try:
        with threadpool_limits(limits=args.threads):
            run_experiment(args)
    except (ValueError, OSError, MemoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
