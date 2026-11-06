# Running the development search

Use the project environment from the repository root:

```sh
uv sync --locked
uv run python scripts/run_pipeline.py develop --config path/to/pipeline.yaml --seed 42
```

Supply your UTF-8 YAML configuration using the [configuration contract](ARCH.md#cli--configuration-contract). Omitting `--seed` preserves the configuration's seed, which defaults to 42. Inputs must already exist at the configured paths. Each invocation creates a fresh run; partial runs cannot be resumed or overwritten.

The command writes one JSON result to stdout and diagnostics to stderr. A failed search exits nonzero. Its run ID locates the terminal trial ledger under `reports/<run_id>/trials.jsonl`, when publication was possible; partial artifacts alone do not indicate success.

During the current [M7 task-1 migration](ARCH.md#development-search-interface-and-staged-cli-migration), a successful search returns `search_complete_pending_freeze`. It has not frozen an experiment and cannot authorize final evaluation. The next [plan task](../PLAN.md#7-orchestrate-development-and-freeze-selection) adds that authorization. See [history](HISTORY.md#m7-task-1--develop-cli-and-final-model-search) for actual verification and limitations.
