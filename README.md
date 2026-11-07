# Ukrainian semi-supervised CRF POS tagger

A Ukrainian part-of-speech tagger that tests whether automatically tagged books can improve a model trained on human-tagged examples. A **tag** is a grammatical category: in `Кіт спить .`, the categories are `NOUN VERB PUNCT`.

The program trains a Conditional Random Field (**CRF**): a model that considers word clues and the sequence of categories in a sentence. **Semi-supervised learning** means combining human-tagged examples with additional examples tagged by the model itself.

For current completion status and remaining real-corpus gates, see [PLAN.md](PLAN.md). Small-fixture acceptance does not establish real-data accuracy. [ARCH.md](docs/ARCH.md) owns the technical contracts; this README is a practical usage guide. Dated verification evidence belongs in [HISTORY.md](docs/HISTORY.md).

## Contents

- [Setup](#setup)
- [Prepare the data](#prepare-the-data)
- [Create experiment.yml](#create-experimentyml)
- [Run the experiment](#run-the-experiment)
- [Read the results](#read-the-results)
- [Tag your own text](#tag-your-own-text)
- [Outputs](#outputs)
- [Project structure and file guide](#project-structure-and-file-guide)
- [Verification commands](#verification-commands)
- [Troubleshooting](#troubleshooting)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then open a terminal in this repository:

```bash
cd /path/to/postag
uv sync --locked
```

Replace `/path/to/postag` with your checkout location. Run all commands below from that root. `uv` manages the local `.venv`; `.python-version` selects CPython 3.14.6 and `uv.lock` records dependency versions. No GPU, spaCy model, or paid service is required. Training uses the CPU; runtime depends on the actual inputs and settings.

Check the available interfaces:

```bash
uv run python scripts/run_pipeline.py --help
uv run python scripts/run_pipeline.py develop --help
uv run python scripts/run_pipeline.py evaluate --help
uv run python scripts/predict.py --help
```

For IDE setup, see [PyCharm guidance](docs/PYCHARM.md).

## Prepare the data

### Understand the separate inputs

| Input | Analogy | Use |
|---|---|---|
| `train_gold` | Worked examples with correct answers | Teach the model |
| `val_gold` | Practice exam | Choose settings |
| `test_gold` | Final exam | Measure the frozen experiment |
| Literature | Unanswered exercises | Supply extra examples for automatic tagging |

**Gold** means human-annotated data. A **token** is one modeled word or punctuation item. Do not combine or reshuffle the official gold splits. Test scores must not guide later setting selection.

### Arrange the files

```text
data/raw/
├── gold/
│   ├── uk_iu-ud-train.conllu
│   ├── uk_iu-ud-dev.conllu
│   └── uk_iu-ud-test.conllu
└── literature/
    ├── book_01.txt
    └── book_02.txt
```

Create missing directories:

```bash
mkdir -p data/raw/gold data/raw/literature
```

Download the three `.conllu` files from one fixed official release of [UD Ukrainian-IU](https://github.com/UniversalDependencies/UD_Ukrainian-IU). Record its actual release tag in the configuration. The upstream `dev` file is this project's `val_gold`. CoNLL-U is a tabular format containing words and linguistic annotations; retain original bytes and official membership.

Place eligible Ukrainian books directly in `literature/`. Discovery accepts regular files with the exact `.txt` suffix, without recursion or symlinks. Files directly in `data/raw/` are not automatically used. New checkouts must supply their own data.

### Complete corpus readiness before real training

Follow the [corpus-readiness gate](PLAN.md#corpus-readiness-and-experiment-gate):

1. Confirm all three gold files belong to the recorded release and preserve the originals. The run records checksums automatically.
2. Confirm the books decode correctly using one configured encoding. The example uses UTF-8. Unusual filenames do not prove the contents are corrupted.
3. Establish literature source provenance and exclude books known to overlap validation/test sources. Exact sentence filtering cannot detect every revised passage or shared source.
4. Check production tokenizer compatibility against reconstructed **training** text, recording limitations without using test labels for tuning. The paragraph preview below is only an initial visual aid, not a corpus compatibility audit.

```bash
uv run python scripts/preview_tokenization.py --text "Її сім’я живе в Києві."
```

There is no single corpus-readiness CLI command. Explicit known-source exclusion rules are available through the Python API's `known_source_exclusions`, not a YAML field. For CLI use, prepare an eligible literature folder excluding known overlapping sources before running. Keep originals unchanged; encoding conversions or other preparations must use separate derived copies and be documented. Do not overwrite existing artifacts or original datasets.

The pipeline separately removes exact gold-text overlaps and duplicate literature sentences, and applies its documented training-derived multiword-token exclusion policy. See [contamination requirements](docs/ARCH.md#strict-test-data-isolation--leakage-prevention).

## Create experiment.yml

Create `experiment.yml` in the repository root. This is a user-created configuration, not a required pre-existing file. `.yaml` also works when passed explicitly with `--config`.

The following example exposes the supported sections and their default values. Replace the release placeholder and supply matching files before running. The authoritative [configuration contract](docs/ARCH.md#cli--configuration-contract) defines validation and compatibility rules.

```yaml
config_version: 1
seed: 42

data:
  train_gold: data/raw/gold/uk_iu-ud-train.conllu
  val_gold: data/raw/gold/uk_iu-ud-dev.conllu
  test_gold: data/raw/gold/uk_iu-ud-test.conllu
  ud_release: "REPLACE_WITH_ACTUAL_RELEASE_TAG"
  literature_dir: data/raw/literature
  source_encoding: utf-8

tag_mapping: {}

preprocessing:
  normalizer_version: 1
  tokenizer_adapter_version: 1

features:
  schema_version: 1

training:
  c1_grid: [0.01, 0.1, 1.0]
  c2_grid: [0.01, 0.1, 1.0]
  parameters:
    max_iterations: 200
    num_memories: 6
    epsilon: 0.00001
    period: 10
    delta: 0.00001
    min_freq: 0.0
    all_possible_states: false
    all_possible_transitions: true

self_training:
  thresholds: [0.90, 0.95, 0.98]
  pseudo_to_gold_ratio: 1.0

resources:
  read_buffer_bytes: 65536
  max_paragraph_chars: 65536
  max_sentence_chars: 16384
  max_sentence_tokens: 256
  batch_tokens: 4096
  memory_budget_mib: 8192

evaluation:
  bootstrap_resamples: 2000
  bootstrap_seed: 42
```

### Options in plain language

| Option | Meaning |
|---|---|
| `config_version` | Configuration format; currently `1` |
| `seed` | Controls seeded development sampling; default `42` |
| `data.train_gold`, `val_gold`, `test_gold` | Three distinct `.conllu` files under `data/raw/gold/` |
| `data.ud_release` | Actual release identifier; required provenance, not a download instruction |
| `data.literature_dir` | Eligible books folder; defaults to `data/raw/literature` |
| `data.source_encoding` | Book decoding, default UTF-8; gold always uses UTF-8 |
| `tag_mapping` | `{}` keeps all 17 original UPOS categories |
| `preprocessing.*_version` | Recorded normalization/tokenizer behavior; supported value `1` |
| `features.schema_version` | Recorded word-clue format; supported value `1` |
| `training.c1_grid`, `c2_grid` | Combinations of regularization strengths to try; regularization discourages overly complicated learning |
| `training.parameters.max_iterations` | Optimization iteration budget; reaching it does not imply convergence |
| `num_memories` | Optimizer history size |
| `epsilon`, `period`, `delta` | Optimizer stopping controls |
| `min_freq` | Feature-frequency cutoff |
| `all_possible_states`, `all_possible_transitions` | Controls expansion of word/tag and adjacent-tag combinations |
| `self_training.thresholds` | Confidence cutoffs to compare using validation |
| `self_training.pseudo_to_gold_ratio` | Maximum automatically tagged words relative to human-tagged training words |
| `resources.read_buffer_bytes` | File-read chunk size |
| `max_paragraph_chars`, `max_sentence_chars` | Raw-text candidate size limits |
| `max_sentence_tokens` | Maximum tokens per raw sentence |
| `batch_tokens` | Inference batch token limit; must be at least `max_sentence_tokens` |
| `memory_budget_mib` | Monitored aggregate memory budget; 8192 MiB is 8 GiB |
| `evaluation.bootstrap_resamples` | Number of resamples for the final uncertainty interval |
| `evaluation.bootstrap_seed` | Separate seed for that final resampling |

`config_version`, `data`, and its gold paths/release are required. Omitted optional sections use defaults. Unknown or duplicate YAML keys fail; do not add arbitrary options. Paths for corpus inputs are repository-relative, regardless of the configuration file's location. Keep the prescribed optimizer defaults for the planned experiment; changing settings changes the experiment you are conducting.

A ratio of `1.0` caps extra tokens at the supervised training token count. A ratio of `0.0` selects no extra tokens and exercises gold-only reuse. Sentences are never split to fill the budget, so the selection can be smaller than the cap.

For a threshold of `0.95`, **every word's predicted tag** must have a marginal score of at least `0.95`; one word below it rejects the whole sentence. These scores are not guaranteed correctness rates or whole-sentence probabilities.

Optional category consolidation replaces `tag_mapping: {}` with:

```yaml
tag_mapping:
  ADP: FUNC
  CCONJ: FUNC
  SCONJ: FUNC
  PART: FUNC
```

This creates a 14-category task. `FUNC` is custom, not standard UD UPOS. Choose the mapping before development; changing it requires a new experiment. See [mapping rules](docs/ARCH.md#optional-target-tag-mapping).

## Run the experiment

### 1. Develop and freeze selection

After corpus readiness:

```bash
uv run python scripts/run_pipeline.py develop --config experiment.yml
```

To override the development seed explicitly:

```bash
uv run python scripts/run_pipeline.py develop --config experiment.yml --seed 42
```

Each invocation starts a **new run**, including repeated commands. Run one of these alternatives for your intended experiment.

The default workflow:

1. Trains nine gold-only candidates and uses validation to select **CRF-Gate**.
2. Tags eligible, contamination-filtered literature with that Gate.
3. Selects complete confident sentences within the token cap for each threshold.
4. Tries nine settings per threshold: 27 final configurations.
5. Selects **CRF-Final** and a gold-only **CRF-GoldControl** with matching regularization settings.
6. Freezes model references, datasets, settings, and completion evidence.

There are 36 planned configurations. Empty extra-data selections reuse corresponding gold-only fits, so actual fit counts can be lower. Gold control comes from retained gold-only candidates; it requires no extra training.

The command writes a JSON result to stdout and diagnostic messages to stderr. Success resembles:

```json
{"status": "complete", "run_id": "YOUR_RUN_ID", "run_path": ".../run.json"}
```

Save the actual run ID. Before evaluation, inspect `reports/YOUR_RUN_ID/run.json`, `trials.jsonl`, `pseudo_selection.json`, and `models/YOUR_RUN_ID/selected.json`. Confirm completed development, full trial accounting, selected pseudo-token counts, resource diagnostics, and selected models. Any further development decisions must use validation, not test results.

### 2. Evaluate the frozen experiment

Replace `YOUR_RUN_ID` in this and subsequent commands:

```bash
uv run python scripts/run_pipeline.py evaluate --run-id YOUR_RUN_ID
```

This compares the three frozen models on the same test examples without retraining. Results appear in `reports/YOUR_RUN_ID/evaluation/`.

An optional `--seed 42` asserts the already-frozen **bootstrap** seed; it does not override it. A repeat evaluation validates and returns a completed cache. Changed inputs/artifacts or partial/failed attempts are rejected; they are not overwritten or automatically resumed.

Do not change model settings in response to these test results and then report the same test as an independent final exam.

## Read the results

| Result | Meaning |
|---|---|
| Token accuracy | Fraction of tokens assigned the correct category |
| Macro-F1 | Average category F1, giving each category equal weight; F1 balances missed and incorrect assignments |
| Per-tag scores / confusion counts | Which categories the model handles well or confuses |
| OOV accuracy | Accuracy on words absent from the human-tagged training vocabulary |
| Non-punctuation accuracy | Accuracy after excluding gold punctuation tokens |
| Final minus Gate | Overall pipeline gain or decline |
| Final minus gold control | Difference against gold-only training at matched regularization |
| Bootstrap interval | Uncertainty estimate for Final-minus-Gate accuracy on evaluated units |

Accuracy values in metric records are fractions in `[0, 1]`; multiply by 100 for percentages. Comparison differences are percentage points. Hypothetically, 95.0% versus 95.4% is a gain of **0.4 percentage points**.

Bootstrap repeatedly samples documents, or sentences when complete document grouping is unavailable, using the same sampled units for both models. An interval crossing zero does not clearly distinguish improvement from evaluation-sample variation. It does not measure variation from retraining. Fewer than two units yields an unavailable interval.

Report negative differences honestly. If no pseudo-labels were selected, reused baseline results do not demonstrate augmentation gains. Benchmark accuracy does not establish accuracy on unannotated books. For report schemas and precise metric definitions, see [evaluation requirements](docs/ARCH.md#evaluation-contract).

## Tag your own text

Find the selected Final candidate ID in `models/YOUR_RUN_ID/selected.json`, then provide its complete bundle directory:

```bash
uv run python scripts/predict.py \
  --model models/YOUR_RUN_ID/candidates/YOUR_CANDIDATE_ID \
  --input /path/to/ukrainian-text.txt
```

Replace both IDs and the input path. Keep `model.joblib` and `metadata.json` together. Only load trusted model bundles. Standalone copies do not require original training corpora, but their recorded Python/dependency environment must be compatible.

Prediction uses the model's recorded encoding and preprocessing. Its JSON result gives the fresh predictions path and counts. Each sentence record includes original forms, predicted labels, confidence scores, and source offsets. Repeated and low-confidence sentences remain in standalone predictions; resource limits still apply. Without human-written answers for your text, this demonstrates tagging rather than measuring accuracy.

## Outputs

`YOUR_RUN_ID` distinguishes one run from another. Generated files are not inputs to edit manually: integrity checks depend on their exact bytes. Do not move or alter inputs used by a frozen experiment before evaluation.

| Location | Contents / purpose |
|---|---|
| `data/interim/YOUR_RUN_ID/` | Disk-backed audit/sampling indexes and intermediate work |
| `data/processed/YOUR_RUN_ID/` | Prepared gold/extended training datasets; JSONL stores one sentence per line |
| `models/YOUR_RUN_ID/candidates/YOUR_CANDIDATE_ID/model.joblib` | Saved fitted model |
| Candidate `metadata.json` | Model settings, environment, identity, checksums, and training evidence |
| `models/YOUR_RUN_ID/selected.json` | Frozen Gate, Final, and gold-control references |
| `reports/YOUR_RUN_ID/effective_config.json` | Full resolved settings |
| `reports/YOUR_RUN_ID/inputs.jsonl` | Original input paths, sizes, encodings, and checksums |
| `reports/YOUR_RUN_ID/trials.jsonl` | Every planned configuration: fitted, reused, failed, or not executed |
| `reports/YOUR_RUN_ID/pseudo_selection.json` | Extra-data selection counts, distributions, exclusions, and cap utilization |
| `reports/YOUR_RUN_ID/run.json` | Development completion or failure descriptor |
| Evaluation `gate.jsonl`, `final.jsonl`, `gold_control.jsonl` | Aligned test predictions |
| Evaluation `metrics.json` | Test scores for all three models |
| Evaluation `comparison.json` | Signed model differences and bootstrap interval |
| Evaluation `units.jsonl` | Membership of resampled documents/sentences |
| Evaluation `diagnostics.json` | Grouping/fallback diagnostics |
| Evaluation `completion.json` | Final evaluation completion and output checksums |
| Evaluation `core.json`, `bootstrap_attempt.json` | Internal integrity and attempt records; not substitutes for completion |
| Standalone prediction's fresh `reports/<prediction_run_id>/` | `predictions.jsonl`, `diagnostics.json`, and `completion.json` |

JSON contains one structured document; JSONL contains one structured record per line. A checksum is a content fingerprint used to detect changes. A directory or model file alone does not prove a successful run. See [artifact contracts](docs/ARCH.md#3-artifact--run-contracts) for authoritative lifecycle details.

## Project structure and file guide

```text
postag/
├── scripts/          # Commands you run
├── src/crf/          # Reusable application code
│   ├── data/         # Read, normalize, split, and filter text
│   ├── features/     # Word clues supplied to the model
│   ├── models/       # Train, score confidence, sample, and save
│   └── utils/        # Configuration, paths, resources, and I/O
├── tests/            # Deterministic software checks
├── docs/             # Specifications and focused guides
├── data/             # Original inputs and separate derived data
├── models/           # Generated model bundles
├── reports/          # Generated run records and scores
├── pyproject.toml
├── uv.lock
└── README.md
```

### Commands and top-level modules

| File | Job |
|---|---|
| `scripts/run_pipeline.py` | `develop` and `evaluate` entry points |
| `scripts/predict.py` | Standalone raw-text tagging |
| `scripts/preview_tokenization.py` | Short paragraph sentence/word preview |
| `scripts/fetch_dictionary.py` | Lexical-resource downloader; not needed by the current surface-feature experiment |
| `src/crf/pipeline.py` | Coordinates development and freezing |
| `src/crf/experiments.py` | Validates saved experiment and selection identities |
| `src/crf/evaluation.py` | Calculates and validates metrics |
| `src/crf/final_evaluation.py` | Frozen three-model test evaluation and cache |
| `src/crf/bootstrap.py` | Paired resampling for uncertainty |
| `src/crf/inference.py` | Streams standalone raw prediction |
| `src/crf/predictions.py` | Prediction record schema and validation |
| `src/crf/reporting.py` | Shared observation/failure record validation |

### Data, features, and models

Paths below are relative to `src/crf/`.

| File | Job |
|---|---|
| `data/conllu.py` | Reads gold annotation and separate label-free audit records |
| `data/literature.py` | Bounded literature discovery and streaming |
| `data/tokenization.py` | Basic Ukrainian tokenizer backend |
| `data/segmentation.py` | Production UD-oriented sentence/token adaptation |
| `data/normalization.py` | Consistent Unicode/text forms with source alignment |
| `data/contamination.py` | Gold overlaps, duplicates, source and multiword-token exclusions |
| `data/identity.py` | Input identities and checksums |
| `data/labels.py` | Category inventory and optional mapping |
| `data/records.py` | Typed sentence records and validation |
| `features/surface.py` | Word, affix, shape, and nearby-word clues |
| `features/schema.py` | Permitted feature names, types, and consistency |
| `models/training.py` | Single-candidate native CRF fitting |
| `models/search.py` | Gold-only candidate search and Gate selection |
| `models/confidence.py` | Predicted-label marginal scores and sentence acceptance |
| `models/sampling.py` | Seeded whole-sentence selection within the cap |
| `models/selection_diagnostics.py` | Selection distributions, counts, and acceptance coverage |
| `models/reuse.py` | Checked baseline reuse for empty selections |
| `models/bundles.py` | Model persistence and loading checks |
| `utils/config.py` | Strict YAML parsing and defaults |
| `utils/paths.py` | Safe source/output path validation |
| `utils/artifacts.py` | Fresh output allocation and publication |
| `utils/resources.py` | Aggregate process memory monitoring |
| `utils/serialization.py` | Strict JSON/JSONL handling |
| `utils/logging.py` | Diagnostic log configuration |
| `utils/dictionary.py` | Reusable lexical-resource download logic |

### Configuration, documentation, and generated support files

| File / directory | Job |
|---|---|
| `pyproject.toml` | Dependencies, package build, and tool configuration |
| `uv.lock` | Exact reproducible dependency resolution |
| `.python-version` | Selected interpreter |
| `.gitignore` | Generated-artifact and cache exclusions |
| `AGENTS.md` | Repository operating procedures |
| `PLAN.md` | Goals, acceptance, and current task status |
| `docs/ARCH.md` | Authoritative architecture, contracts, and design decisions |
| `docs/HISTORY.md` | Dated changes, verification evidence, and limitations |
| `docs/DEVELOPMENT.md` | Focused CLI usage |
| `docs/TESTING.md` | Verification dependencies and fixture limitations |
| `docs/PYCHARM.md` | IDE setup |
| `docs/TOKENIZATION.md` | Low-level preview guide; historical future-adapter wording is superseded by ARCH/PLAN |
| `tests/` | Small fixtures and boundary/integration checks; `test_end_to_end.py` exercises the complete CLI workflow |
| `.venv/` | Project-local Python environment |
| `dist/` | Built installable packages, when generated |
| `__init__.py`, `py.typed` | Python package and typing markers |
| `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` | Automatically generated caches |

## Verification commands

Run from the repository root:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/crf/
uv run pytest tests/ -v
```

For a focused complete-workflow smoke check:

```bash
uv run pytest tests/test_end_to_end.py -v
```

For separately marked integration tests:

```bash
uv run pytest tests/ -m integration -v
```

Tests use small deterministic fixtures, not production books or official-corpus downloads. Actual CLI/RSS integration checks require local process-inspection permission. See [testing guidance](docs/TESTING.md). Passing checks verifies software behavior; it does not establish an accuracy gain on real data.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing gold split or no eligible literature | Check configuration paths and direct `.txt` children; never substitute another gold split |
| Unicode decoding error | Identify the actual book encoding; prepare separate consistent copies without altering originals |
| Unsupported/duplicate YAML setting | Use the documented keys once each; check spelling and indentation |
| No extra sentences selected | Inspect confidence and cap diagnostics; empty selections legitimately reuse baselines |
| Memory-budget failure | Inspect diagnostics and plan a new run with an appropriate budget/data cap before test evaluation; monitoring is best effort, not an OS limit |
| Process-inspection permission denied | Use an execution context that allows local process monitoring; do not disable the safeguard |
| Bundle environment mismatch | Restore the recorded compatible Python/dependency environment |
| Changed checksum or incomplete experiment | Preserve the failed evidence and investigate; do not edit completion files or force evaluation |
| Interrupted development/evaluation | Partial runs are not automatically resumed or overwritten; see lifecycle contracts before arranging another attempt |

Keep generated models and datasets out of commits. Preserve the frozen run and linked inputs to retain reproducible evaluation evidence.
