# Feature vectors and HDBSCAN experiments

All shell commands and Python examples below run from the repository root.

## Quick start

The existing `data/tokenized/` files are the input. You do not need to retokenize.
The dependencies below were tested in this project's Python 3.9.6 environment.

```bash
source .venv/bin/activate
python -m pip install -r requirements/experiment.txt
python scripts/fetch_dictionary.py
python -m postag.experiment --sample-size 6000 --output runs/my-experiment
```

Read `runs/my-experiment/report.md`, then follow its cluster links or open a
configuration's `inspection.md`. `results.json` contains the complete metrics.
Use a different output directory for every experiment; no overwrite option is
provided. Without `--output`, a timestamped directory is created under `runs/`.
Defaults are anchored to the repository root; explicit relative paths use the current
working directory. Successful runs return exit code 0; invalid data/configuration
returns a nonzero code. A failed run's manifest records the error and its completed
artifacts remain available.

The default dictionary is used automatically when present. If it has not been
downloaded, clustering still runs and the report explicitly marks POS evaluation
pending. `--no-dictionary` skips evaluation deliberately. Explicit dictionary paths
must exist and parse correctly; invalid resources are not silently ignored.

## What each module does

| File | Responsibility |
| --- | --- |
| `postag/tokenization.py` | Existing sentence/token generation, unchanged by this implementation. |
| `postag/features.py` | `token_features`, `word2features`, `sent2features`, and TF helper functions; no training execution. |
| `postag/corpus.py` | Validated corpus loading, full-corpus TF, file hashes, occurrence sampling and provenance. |
| `postag/vectors.py` | Feature selection, sparse `DictVectorizer`, optional numeric scaling, SVD and normalization. |
| `postag/evaluation.py` | Dictionary parsing, ambiguity/coverage, partition metrics and cluster examples. |
| `postag/experiment.py` | CLI, reproducible parameter sweep, saved artifacts and reports. |
| `scripts/fetch_dictionary.py` | Download/verify the pinned prebuilt dictionary from `data/dict_uk/source.json`. |

## Features and sampling

Each row represents one token occurrence. Repeated words retain different neighbor
contexts. Original capitalization, punctuation, and token order are retained.
Only derived affixes, word identity, and frequency lookup keys are lowercase.
Case/digit flags use the original string. There is no constant bias column.

All positions share prefix lengths 1–3, suffix lengths 1–4, character length,
`istitle`, `isupper`, and `isdigit`. BOS/EOS flag sentence edges. Affixes of short
tokens are ordinary Python slices; long requested slices return the entire token.
`isdigit` is not a decimal-number parser. `istitle` follows Python's title-case
definition, rather than merely testing the first character.

| Variant | Token features | Neighbor features | TF | Full lowercase word identity |
| --- | --- | --- | --- | --- |
| `token` | Yes | No | No | No |
| `context` | Yes | Yes | No | No |
| `context_tf` | Yes | Yes | Yes, at all three positions | No |
| `context_tf_identity` | Yes | Yes | Yes, at all three positions | Yes, at all three positions |

The first three are compared by default. Word identity is optional because it
can encourage clusters of repeated forms. Edit `PREFIX_LENGTHS`, `SUFFIX_LENGTHS`,
and `WORD_FEATURES` in `postag/features.py` to extend the shared features. Code hashes and
feature names are recorded with the run.

TF is `count(lower(token)) / total tokens` over the entire input corpus, including
numbers, punctuation, and standalone artifacts. Two corpus passes first count TF
and then extract the selected features. Only one document is parsed at a time,
although the TF vocabulary and sampled feature dictionaries stay in memory.
Input files are hashed and checked again on the second pass.

The default 10,000-token sample uses approximately equal quotas per document,
redistributing unused quotas from short documents. Within each document, seeded
sampling chooses token occurrences uniformly without replacement. This makes it
easy to sample across all documents but gives short documents more weight per
token. It is not a uniformly sampled estimate of whole-corpus performance.
Rows are saved in document/sentence/token order, never sampling order.

`--rows all` includes every token kind in sampling and clustering. `--rows lexical`
selects only tokens containing at least one Unicode letter. Both keep punctuation
and all other tokens in neighbor context; neither edits the tokenizer output.
POS scoring always uses only lexical rows with one dictionary coarse POS.
Standalone BOMs and symbols without letters have kind `other`; numbers and pure
punctuation have separate kinds. Decimal tokens without letters are nonlexical too.

## Experiment settings

```bash
# Shorter single-configuration run.
python -m postag.experiment --output runs/small --sample-size 2000 \
  --variants context_tf --components 30 --min-cluster-sizes 15 --min-samples 5

# Compare word identity with the main feature set.
python -m postag.experiment --output runs/identity --sample-size 6000 \
  --variants context_tf context_tf_identity

# Assess numeric scaling on the same seeded rows as an earlier run.
python -m postag.experiment --output runs/scaled --sample-size 6000 --numeric-scaling std

# Cluster lexical tokens, retaining all original neighbors.
python -m postag.experiment --output runs/lexical --sample-size 6000 --rows lexical

# Use a custom expanded export and its optional provenance manifest.
python -m postag.experiment --output runs/custom --dictionary /path/dict_corp_lt.txt \
  --dictionary-format lt --dictionary-source /path/source.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | `data/tokenized/` under the repository root | Directory of tokenizer JSON files; recursive, sorted discovery. |
| `--sample-size` | `10000` | Maximum selected occurrences; `0` explicitly requests all eligible occurrences. |
| `--rows` | `all` | `all` or `lexical` target rows; context always uses original tokens. |
| `--seed` | `42` | Document sampling and randomized SVD seed. |
| `--variants` | `token context context_tf` | One or more feature sets from the table above. |
| `--components` | `30 50` | Requested SVD dimensions; capped to `min(rows, columns) - 1` on small inputs. |
| `--min-cluster-sizes` | `15 40` | HDBSCAN minimum cluster sizes, each at least 2. |
| `--min-samples` | `5 10` | Density settings, including the point itself in scikit-learn's convention. |
| `--numeric-scaling` | `none` | `std` divides only length and TF columns by their sampled standard deviations; no centering. |
| `--normalize` | Off | Apply L2 normalization after SVD. |
| `--threads` | `1` | BLAS/OpenMP thread limit; HDBSCAN uses one job. |

The CLI runs the Cartesian product of variants, component counts, cluster sizes,
and density settings. The default is 24 configurations. Parameters exceeding the
sample size are recorded as skipped. Fewer than three selected rows is an error.
Small inputs can cap both requested SVD dimensions to the same actual dimension;
the manifest records both, so those runs should not be interpreted as independent
dimensionality comparisons.

The sparse matrix is never densified before SVD. Numeric ranges are recorded to
show whether length dominates the unscaled baseline or TF is comparatively small.
Scaling is an experiment choice, not an automatic claim of improvement.
HDBSCAN uses `sklearn.cluster.HDBSCAN` with Euclidean distance, KD-tree computation,
EOM selection, and `allow_single_cluster=False`. It does not use the separate
`hdbscan` package. A result can still have one cluster plus noise, or all noise.

SVD numerical warnings are captured in the manifest and report. The code checks
that reduced vectors, components, and singular values are finite and that the
components are orthonormal. Numerical warnings were observed with this machine's
NumPy/OpenBLAS setup; similar macOS matmul warning behavior is tracked in
[NumPy issue 28687](https://github.com/numpy/numpy/issues/28687).
The warnings are retained rather than globally suppressed. The pilot artifacts
were also independently reloaded and compared against recomputed vectors.

Record a bounded pilot before requesting the full corpus. The report contains
sparse/reduced matrix storage and a full-corpus dense-vector lower bound. These
do not include Python feature dictionaries, SVD workspace, dictionary parsing,
or HDBSCAN working memory. Runtime does not scale linearly with token count.

## Dictionary source and scoring policy

The pinned resource is the prebuilt `dict_corp_vis.txt.bz2` from
[dict_uk v6.8.6](https://github.com/brown-uk/dict_uk/releases/tag/v6.8.6).
It contains inflected forms, including lemma forms. The downloader verifies the
SHA-256 recorded in `data/dict_uk/source.json`, and the experiment verifies it again.
No Java/Gradle build is needed. The data are attributed to Andriy Rysin, Vasyl
Starko, and the BrUK team and licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as stated by
the [dictionary project](https://github.com/brown-uk/dict_uk#license).
The compressed download and generated runs are ignored by Git; source metadata
is kept in the project.

The parser accepts UTF-8 plain or bzip2-compressed exports:

- `vis`: `surface-form tags`, including indented inflections and `#` comments.
- `lt`: `surface-form lemma tags`, the flat three-column export.

Malformed entries and unknown base POS tags stop evaluation with a line number.
It scans all analyses even though only sampled forms are stored. Additional
morphological analyses with the same coarse POS do not create POS ambiguity.

Lookup lowercases and maps curly/modifier apostrophes (`’`, `ʼ`) to ASCII `'` on
both sides. All resulting analyses are unioned. This deliberately treats case
collisions conservatively rather than inventing a contextual disambiguation rule.
Original tokens, feature affixes, and TF keys are not changed by this lookup policy.

Coarse categories follow the
[VESUM tag definitions](https://github.com/brown-uk/dict_uk/blob/v6.8.6/doc/tags.txt):

| Dictionary tag | Evaluation class |
| --- | --- |
| Any `:pron` analysis | `PRON` |
| `noun` with `:prop` | `PROPN` |
| `adj` with `:numr` | `NUM` |
| `noun`, `adj`, `verb`, `adv` | `NOUN`, `ADJ`, `VERB`, `ADV` |
| `advp` | `ADVP` (adverbial participle) |
| `prep`, `conj`, `part`, `intj` | `PREP`, `CONJ`, `PART`, `INTJ` |
| `numr`, `number` | `NUM` |
| `noninfl`, `onomat` | `NONINFL`, `ONOMAT` |

Overrides are applied in the order shown. Participles with `adj:...:adjp` remain
`ADJ`. This is a coarse VESUM grouping, not a Universal Dependencies conversion.
Style/rarity flags do not discard analyses. Every dictionary analysis remains
eligible to contribute ambiguity.

Reference statuses are `eligible` (exactly one coarse POS), `ambiguous`, `unmatched`,
or `nonlexical`. Reports include occurrence counts and distinct normalized-form
counts, dictionary coverage, and unambiguous coverage. Repeated occurrences affect
the metrics independently; scores are not type-balanced.

Primary V-measure, ARI, and purity score eligible **non-noise** occurrences.
Noise label `-1` is not counted as a discovered cluster. Scores are accompanied
by eligible/assigned counts and noise fraction. Empty evaluations are `null` with
a reason. Single-class/single-cluster cases use scikit-learn's metric conventions;
high values on a trivial subset should not be taken as useful POS separation.

`results.json` also includes:

- Scores on all eligible rows treating noise as one label, explicitly secondary.
- A one-cluster reference on all eligible rows and each run's non-noise subset.
- Scores on the common non-noise eligible rows across all successful configurations.
  The intersection may be small or empty; its row IDs are saved separately.

Purity is the sum of each cluster's largest POS count divided by scored occurrences.
It can look good with many small clusters. Compare it with V-measure, ARI, cluster
counts, and coverage. No cluster-to-POS renaming is required for these partition
metrics; majority POS is supplied only for inspection.

## Reloading and reusing the vectors

Each run has one shared `rows.jsonl` and `features.jsonl.gz`. All matrices and
assignment arrays use that exact row order. `row_id` encodes the document-relative
path and zero-based sentence/token indices. Context snippets contain up to eight
tokens on either side; use the source JSON and indices for the full sentence.

```python
import json
from pathlib import Path
from postag.artifacts import load_transform
import numpy as np
from scipy.sparse import load_npz
from postag.features import sent2features

run = Path("runs/my-experiment")
variant = run / "context_tf"
rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
X_sparse = load_npz(variant / "features.npz")
X_reduced = np.load(variant / "svd-30/vectors.npy", allow_pickle=False)
assert X_sparse.shape[0] == X_reduced.shape[0] == len(rows)

# Load only locally generated/trusted joblib artifacts.
encoder = load_transform(variant / "encoder.joblib")
reducer = load_transform(variant / "svd-30/reducer.joblib")
tf = json.loads((run / "word_frequencies.json").read_text())
new_features = sent2features(["Олена", "живе", "в", "Києві", "."], tf)
new_vectors = reducer.transform(encoder.transform(new_features))
```

The compatibility loader also supports the original `vectors.FeatureEncoder`
module path in existing runs without modifying those files. New artifacts use
`postag.vectors.FeatureEncoder`.

The encoder reuses the fitted vocabulary and numeric scales; the reducer reuses
SVD and any L2 normalization. New categorical values absent during fitting have
no vocabulary column; unseen TF entries are zero. This produces vectors, not
predicted POS labels. The chosen HDBSCAN experiment does not implement assignment
of future tokens to existing clusters.

## Validation and interpretation

```bash
python -B -m unittest discover -s tests -v
```

Tests cover original tokenizer behavior, feature semantics, TF and sampling,
row/context alignment, changed-input detection, sparse transform persistence,
unknown categorical values, small-input SVD, dictionary ambiguity and checksum
validation, hand-checkable metrics/noise, end-to-end artifacts, and overwrite
protection. The tiny dictionary in tests is synthetic, not the evaluation corpus.

Inspect clusters for adjective endings `-ий/-ій`, infinitives ending in `-ти`,
mixed POS, repeated function words, punctuation, and noise. Reports include
ending counts but do not infer POS from those endings. A dictionary cannot
identify which analysis an ambiguous occurrence takes in context. Good scores
on this selected dictionary subset are evidence for further study, not a trained
POS tagger or contextual gold-standard accuracy. Parameter/feature selection on
these same scores is exploratory. Held-out work would fit TF, vocabulary, scaling,
and SVD on training data only.

Method references: [DictVectorizer](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.DictVectorizer.html),
[TruncatedSVD](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.TruncatedSVD.html),
[HDBSCAN](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.HDBSCAN.html).
