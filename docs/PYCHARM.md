# PyCharm and the uv environment

Run `uv sync --locked` from the repository root. The project uses the existing
`.venv/bin/python` interpreter selected by `.python-version` (CPython 3.14.6).
Dependencies and development tools come from `pyproject.toml` and `uv.lock`.

The local PyCharm SDK is named `Python 3.14 (postag, uv)`. Interpreter registrations
are local IDE settings, so another checkout/machine needs its own registration:

1. Open Settings → Python → Interpreter → Add Interpreter → Add Local Interpreter.
2. Select the existing **uv** environment, rather than generating a replacement.
3. If detection fails, set the uv executable to the result of `command -v uv`.
4. Select this repository's `.venv/bin/python` and name the SDK
   `Python 3.14 (postag, uv)`.

See [JetBrains' uv setup instructions](https://www.jetbrains.com/help/pycharm/uv.html).
After externally changing IDE settings, restart PyCharm so it reloads the SDK
registry, then reopen the project. Allow interpreter skeletons/indexing to refresh.
The interpreter selector should show Python 3.14 with uv, rather than the old 3.9 SDK.

The module inherits the project interpreter, marks `src` as sources and `tests`
as test sources, and excludes the environment, caches, distributions and generated
run directories from indexing. The automatic pyproject module model is disabled: during setup it regenerated
incorrect `.idea/src` roots. Native uv still manages dependencies from
`pyproject.toml` and `uv.lock`; this setting controls only IDE module generation.

Shared run configurations **Tests** and **Tokenizer preview** use the project
interpreter and repository working directory. They do not inject source/content
roots into `PYTHONPATH`; imports use the package installed by uv. Pytest is the
default test runner. Equivalent commands are:

```bash
uv run python -m pytest tests/ -v
uv run python scripts/preview_tokenization.py --text "Це проф. Петренко. Він у Києві."
```

Use uv to change dependencies and commit both declarations and lockfile. Keep
Ruff/mypy settings in `pyproject.toml`; the obsolete Black interpreter setting
has been removed. No separate pip-managed or IDE-created environment is needed.
