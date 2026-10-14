# Ukrainian token features and POS clustering

Tokenize Ukrainian texts, build numerical token features, and investigate POS
structure with SVD, HDBSCAN, and dictionary-based evaluation.

## Directory map

| Directory | Purpose |
| --- | --- |
| [postag/](postag/README.md) | Python package: tokenization, features, vectors, clustering, and evaluation. |
| [scripts/](scripts/README.md) | Dictionary acquisition and maintenance commands. |
| [data/](data/README.md) | Raw texts, tokenized corpus, sample text, and reference dictionary. |
| [requirements/](requirements/README.md) | Pinned dependencies for tokenization and experiments. |
| [tests/](tests/README.md) | Automated unit and integration checks. |
| [docs/](docs/README.md) | Detailed guides, implementation plans, and measured results. |
| [runs/](runs/README.md) | Generated experiment artifacts and inspection reports. |

## Run an experiment

Run commands from the repository root. Use the existing virtual environment or
create one with `python3 -m venv .venv`.

```bash
source .venv/bin/activate
python -m pip install -r requirements/experiment.txt
python scripts/fetch_dictionary.py
python -m postag.experiment --sample-size 6000 --output runs/my-experiment
```

The saved corpus is already in `data/tokenized/`. Each experiment creates a new
output directory with its own README and `report.md`. Use a new directory name
for each run. Dictionary download is pinned and checksum-verified; subsequent
runs work offline. Use `--no-dictionary` for vector/clustering experiments alone.

## Tokenize new text

```bash
python -m pip install -r requirements/tokenization.txt
python -m postag.tokenization --text "Олена живе в Києві."
```

With no arguments, the tokenizer reads `data/raw/` and writes `data/tokenized/`.
It preserves capitalization and punctuation and protects existing output files.
See the [tokenization guide](docs/TOKENIZATION.md) before regenerating the corpus.

## Check the project

```bash
python -B -m unittest discover -s tests -v
```

Read the [experiment guide](docs/EXPERIMENTS.md) for feature configuration and
saved-vector reuse, and [initial results](docs/RESULTS.md) for the measured
comparison. The [implementation plan](docs/PLAN.md) records the current scope.

## Changes from the original flat layout

Python modules now live in `postag/`; `main.py` is named `postag/features.py`.
Use `python -m postag.experiment` and `python -m postag.tokenization` instead of
executing root-level scripts. Raw texts moved from `ukrainian_200texts/` to
`data/raw/`, and tokenized texts moved to `data/tokenized/`.

Existing run artifacts are preserved. Load their fitted transforms through
`postag.artifacts.load_transform`, which understands both old and new module
paths. Historical manifests retain the original run's paths and code hashes.
