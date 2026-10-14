# Initial corpus experiment results

All shell commands and Python examples below run from the repository root.

The pipeline successfully produces reusable token vectors and evaluates them with HDBSCAN. This first experiment shows partial POS structure, substantial noise, and many repeated-word or punctuation clusters. It does not yet establish reliable POS tagging.

## Reproduce

```bash
.venv/bin/python scripts/fetch_dictionary.py
.venv/bin/python -m postag.experiment --sample-size 6000 --output runs/baseline
.venv/bin/python -m postag.experiment --sample-size 6000 --output runs/scaled --numeric-scaling std
```

These output directories already exist locally. Choose new names to rerun. All dependencies were already installed; their tested versions are pinned in `requirements/experiment.txt`. Both runs use seed 42, all-token sampling, 30/50 SVD components, minimum cluster sizes 15/40, and min_samples 5/10. Each runs 24 configurations.

Full reports: [baseline](../runs/baseline/report.md) and [scaled](../runs/scaled/report.md). Saved matrices, transforms, row metadata, cluster assignments, and inspection files are inside each run. Run artifacts are ignored by Git and can be regenerated.

## Corpus and dictionary coverage

- 10,258,350 tokens across 200 tokenized documents; TF counts the full corpus.
- Exactly the same 6,000 token occurrences are used in both runs, sampled across all 200 documents.
- 4,774 sampled occurrences are lexical; 1,226 are numbers, punctuation, or other nonlexical tokens.
- Dictionary: VESUM/dict_uk release v6.8.6, verified against the pinned SHA-256 in `data/dict_uk/source.json`.
- 3,140 lexical occurrences have one coarse POS; 1,493 are ambiguous and 141 unmatched.
- Dictionary coverage: 97.05% of lexical occurrences; unambiguous coverage: 65.77%.
- Of 2,772 distinct normalized lexical forms, 2,109 are unambiguous, 530 ambiguous, and 133 unmatched.

## Best observed configuration per feature variant

Each row below selects the highest non-noise V-measure within that variant and scaling choice. The scored subsets differ because HDBSCAN assigns different rows to noise. These are exploratory comparisons, not held-out results or a definitive model ranking.

| Numeric scaling | Variant | SVD / cluster size / min_samples | Clusters | Eligible assigned / 3,140 | Eligible noise | V-measure | ARI | Purity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | token | 50 / 15 / 10 | 61 | 1,066 | 66.1% | 0.4975 | 0.2434 | 0.7542 |
| none | context | 50 / 15 / 5 | 25 | 1,437 | 54.2% | 0.1983 | 0.0493 | 0.4175 |
| none | context_tf | 50 / 15 / 5 | 25 | 1,437 | 54.2% | 0.1983 | 0.0493 | 0.4175 |
| std | token | 50 / 15 / 10 | 84 | 1,618 | 48.5% | 0.4026 | 0.1335 | 0.7040 |
| std | context | 50 / 15 / 5 | 32 | 2,118 | 32.5% | 0.1948 | 0.0280 | 0.4046 |
| std | context_tf | 50 / 15 / 5 | 29 | 2,341 | 25.4% | 0.1799 | 0.0266 | 0.3853 |

The unscaled token-only result has the highest observed V-measure (0.4975), but scores only 1,066 of the 3,140 eligible occurrences. Its all-eligible score, treating noise as one label, is V-measure 0.2984 and ARI 0.0768. The single-cluster reference on all eligible rows has V-measure 0, ARI 0, and purity 0.2548.

Unscaled context and context-plus-TF have identical best-result metrics here; raw TF is small compared with length and categorical features. Numeric scaling increases assigned coverage in the displayed results, while their non-noise POS scores do not improve. This supports keeping scaling explicit and inspecting coverage rather than selecting by purity alone.

`results.json` also scores the common non-noise eligible intersection within each sweep: 203 rows for the baseline and 529 for the scaled sweep. Those intersections differ between sweeps and should not be compared as if they were the same population.

## Manual cluster inspection

Inspection of the unscaled `token/svd-50/mcs-15-ms-10` result found:

- Cluster 0 contains 533 commas; cluster 12 contains 283 periods. These do not enter lexical POS scoring.
- Cluster 16 contains 122 occurrences of `не`/`Не`, illustrating repeated-word grouping.
- Cluster 34 contains 15 occurrences of `ти`/`Ти`, all unambiguous PRON. Its `-ти` ending is not evidence of an infinitive cluster.
- Cluster 46 contains 109 occurrences and mixes nouns, verbs, adverbs, adjectives, and other classes; examples include `навіть`, `людини`, `хочете`, and `шукати`.
- Clean adjective/infinitive groupings were not established by these inspected clusters. Suffix counts alone are not POS labels.

Examples and sentence windows: [token-only inspection](../runs/baseline/token/svd-50/mcs-15-ms-10/inspection.md) and [context-plus-TF inspection](../runs/baseline/context_tf/svd-50/mcs-15-ms-5/inspection.md). Full source positions and all POS distributions are in the adjacent `clusters.json` files.

## Resources and validation

The baseline took 51.0 seconds and the scaled sweep 55.1 seconds on this machine, including corpus passes, dictionary lookup, vectorization, clustering, and artifact output. Timings are local observations, not performance guarantees.

The baseline sparse matrices are 6,000 × 5,402 (token), 6,000 × 15,927 (context), and 6,000 × 15,930 (context-plus-TF). Their CSR storage is approximately 0.83, 2.29, and 2.50 MB. Each 50-dimensional reduced matrix takes 2.4 MB. Full-corpus 50-dimensional vectors alone would require about 4.10 GB, before feature dictionaries or algorithm workspace. Full-corpus HDBSCAN was not attempted.

- All 25 automated tests passed, including the unchanged tokenizer suite and end-to-end artifact tests.
- Both real runs were independently reloaded: sparse matrices matched the fitted encoder output exactly, and all 12 reduced matrices matched recomputed transforms exactly.
- All 48 assignment files matched the saved row IDs and had finite membership strengths.
- Stored code hashes matched the implemented modules; baseline and scaled samples had identical row metadata.
- NumPy/OpenBLAS emitted SVD matmul warnings. They are preserved in the reports/manifests. Vectors and fitted parameters were finite, and component orthogonality errors were below 2e-14. Warnings were not globally disabled.

The preliminary `runs/pilot/` artifacts are also preserved; the baseline and scaled reports above include the final numerical diagnostics. The artifact audit is saved in `runs/artifact-audit.json`.

## Limits of these results

This is a document-balanced sample, not a uniform whole-corpus estimate. Dictionary ambiguity and missing forms restrict what is scored; repeated words count repeatedly. Many POS classes can share orthographic features, and HDBSCAN can emphasize lexical repetition or punctuation. No contextual gold-standard corpus, held-out feature selection, full-corpus clustering, or supervised POS model is claimed. The delivered pipeline makes these effects measurable and lets subsequent feature changes be compared reproducibly.

## Directory reorganization

The commands above use the current package layout. Original run manifests retain
their historical paths and code hashes; `tokenized/` moved to `data/tokenized/`,
and Python modules moved to `postag/` (`main.py` became `features.py`). No existing
vectors, encoders, assignments, dictionary, or corpus contents were rewritten.
Use `postag.artifacts.load_transform` to load the original encoders.
