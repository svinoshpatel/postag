# Local verification

Run from the repository root using the locked project environment (`uv sync --locked`). The selected runtime and dependency contract are in [ARCH.md](ARCH.md#4-environment--packaging-contract); the four mandatory commands are in [AGENTS.md](../AGENTS.md#required-checks). Actual dated results belong in [HISTORY.md](HISTORY.md), not this guide.

The complete CLI workflow is exercised with:

```sh
uv run pytest tests/test_end_to_end.py -v
```

These integration cases need the installed project and locked CRFsuite, joblib, NumPy and PyYAML dependencies. They create synthetic CoNLL-U/text inputs and every output in temporary repositories. They do not download data or access production corpora. In-process cases prohibit socket connections, use controlled RSS samples, and force only the confidence branch needed to exercise default augmentation; fitting, serialization, scoring, bootstrap and publication remain real. The zero-budget case exercises explicit baseline reuse and a custom mapped inventory.

The child-process smoke runs the actual develop/evaluate/predict scripts with real process-tree RSS inspection. It requires that local process-enumeration permission, as does the existing real-RSS smoke. A restricted runner that denies it cannot verify that case; use an approved execution context rather than weakening the monitor or reporting the denied check as passed.

Declared integration cases can also be selected with:

```sh
uv run pytest tests/ -m integration -v
```

Focused boundary tests are useful during changes, but do not replace the required full checks. Native imports/fits and permission-dependent monitoring are documented in the corresponding test modules. Synthetic confidence fixtures and iteration-limited tiny fits verify implementation behavior; they establish neither official-corpus compatibility nor real tagging accuracy. Keep the separate [corpus readiness gates](../PLAN.md#corpus-readiness-and-experiment-gate) open until real evidence exists.
