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
