# Running the development search

Use the project environment from the repository root:

```sh
uv sync --locked
uv run python scripts/run_pipeline.py develop --config path/to/pipeline.yaml --seed 42
```

Supply your UTF-8 YAML configuration using the [configuration contract](ARCH.md#cli--configuration-contract). Omitting `--seed` preserves the configuration's seed, which defaults to 42. Inputs must already exist at the configured paths. Each invocation creates a fresh run; partial runs cannot be resumed or overwritten.

The command writes one JSON result to stdout and diagnostics to stderr. A failed search exits nonzero. Its run ID locates the terminal trial ledger under `reports/<run_id>/trials.jsonl`, when publication was possible; partial artifacts alone do not indicate success.

A successful command returns `complete`, with the run ID and path to `run.json`. The [frozen selection](ARCH.md#frozen-model-selection) records the gate, final model and matched gold-only control. A valid completion record and unchanged linked artifacts are required before evaluation; a selection file alone does not authorize it.

The library's `search_development` primitive runs only the search and cannot authorize evaluation. `develop` performs the complete search/freeze workflow; `crf.experiments.load_development` validates an existing completed run without changing its history. See [history](HISTORY.md#m7-task-2--frozen-selection-and-development-authorization) for actual verification and limitations.

For a completed frozen run, evaluate with the recorded settings:

```sh
uv run python scripts/run_pipeline.py evaluate --run-id RUN_ID --seed 42
```

The optional seed asserts the frozen bootstrap seed; it cannot change it. Successful results under `reports/RUN_ID/evaluation/` are validated and returned from cache. Partial or failed attempts are not overwritten or automatically resumed. See the [evaluation contract](ARCH.md#final-evaluation-reports) for report interpretation.

For independent raw text prediction:

```sh
uv run python scripts/predict.py --model models/RUN_ID/candidates/CANDIDATE_ID --input path/to/text.txt
```

Use one trusted bundle directory, including its sidecar. Prediction follows its recorded encoding and preprocessing and creates a fresh report directory. The JSON result identifies the predictions, counts and label space; `diagnostics.json` reports resource exclusions and tokenizer limitations. Copied standalone bundles do not need their original datasets. See [standalone inference](ARCH.md#standalone-inference-interface) for the authoritative behavior.
