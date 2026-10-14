# Ukrainian token feature vectors and clustering plan

All shell commands and Python examples below run from the repository root.

## Goal and scope

Produce reproducible numerical feature vectors for Ukrainian token occurrences,
then use HDBSCAN to investigate whether those vectors separate parts of speech.
Provide inspectable clusters and, where dictionary coverage allows, quantitative
evaluation. Useful POS clusters are an experimental outcome, not a guarantee.
An honest result showing weak clusters or mostly noise still answers the experiment.

This document translates the proposed four steps into English and relates them
to the current project. The checklist below is implemented. Supervised CRF training,
building a production POS tagger, and replacing the tokenizer are outside scope.

## Original proposal in English

1. **Load the data and tokenize correctly.** Download the corpus from the
   supervisor's Google Drive link. Split text into sentences, then split each
   sentence into words and punctuation. Preserve punctuation and original case:
   capitalization is useful for proper names and sentence starts.
2. **Implement feature extraction, similar to `word2features`.** Given a sentence
   and a token index, return a dictionary containing suffixes of lengths 1–4,
   prefixes of lengths 1–3, token length, `istitle`, `isupper`, and `isdigit`.
   Include the same features for the previous and next tokens, plus relative
   corpus frequency (TF), as suggested by the supervisor.
3. **Vectorize and experiment with HDBSCAN.** Convert feature dictionaries into
   numerical vectors with `DictVectorizer`. Reduce the large sparse matrix with
   `TruncatedSVD`, initially to 30–50 components. Run HDBSCAN, count clusters,
   and inspect examples: do adjectives ending in `-ий/-ій` or verbs ending in
   `-ти` group together?
4. **Evaluate against `dict_uk`.** Obtain the Ukrainian dictionary and its POS
   information, compare cluster assignments with dictionary tags, and calculate
   V-measure, purity, and/or Adjusted Rand Index (ARI).

The last step needs one qualification: a dictionary supplies possible analyses
of a word form, not necessarily the correct POS of each occurrence in context.
Treat this as dictionary-based reference evaluation; true contextual gold-standard
evaluation would require a manually annotated corpus.

## Current project status

Implementation is complete for the bounded exploratory experiment. Tokenization
and the original corpus files were left unchanged. The removed CRF exercise was
not restored; `postag/features.py` supplies only reusable feature helpers, including a small
`sent2features` wrapper used by the pipeline tests and documented reuse example.

| Step | Implemented state |
| --- | --- |
| 1. Corpus and tokenization | Existing 200 source files and 200 tokenized JSON files are reused. The original Drive provenance is still unverified. |
| 2. Features | `postag/features.py` and `postag/corpus.py` supply token/context features, full-corpus TF, configurable feature variants, deterministic occurrence sampling, and source/row provenance. |
| 3. Numerical vectors and clustering | `postag/vectors.py` and `postag/experiment.py` persist sparse vectors and fitted transforms, compare SVD/HDBSCAN configurations, and export assignments and inspection reports. |
| 4. Dictionary evaluation | `scripts/fetch_dictionary.py` obtains a pinned, checksum-verified export; `postag/evaluation.py` handles coarse POS mapping, ambiguity, coverage, noise, and clustering metrics. |

Run instructions and design choices are in [EXPERIMENTS.md](EXPERIMENTS.md).
Tests cover the tokenizer, features, sampling/alignment, persisted transforms,
dictionary handling, metrics, and a small end-to-end experiment.
Full-corpus TF was computed, but clustering is intentionally bounded to sampled
occurrences. Full-corpus HDBSCAN and contextual gold-standard POS tagging remain
outside the delivered experiment.

## Implementation checklist

### 1. Connect the existing tokens to feature extraction

- [x] Read the existing JSON format: a list of objects with `sentence` and
  `tokens`. Validate it and process files in deterministic relative-path order.
- [x] Use one row per **token occurrence**, not one row per unique word. Repeated
  words may have different contexts. Retain document path, sentence index, token
  index, original token, and a stable row ID separately from numerical features.
- [x] Reuse the existing helpers, either through imports or a small dedicated
  feature module if separation is useful. Build sentence-level feature rows by
  calling `word2features` for each token index. A small `sent2features` wrapper
  supports sentence-level reuse.
- [x] Keep the existing empty-sentence and annotation-independence checks working
  through the restored minimal `sent2features` wrapper. Remove stale CRF/tutorial
  comments and the constant bias feature.
- [x] Retain original strings and sentence boundaries. Lowercase only derived
  affix/lookup features; compute case flags from the original token. Context must
  never cross sentence or document boundaries.
- [x] Preserve current TF semantics for the baseline:
  `TF(w) = count(lower(w)) / total token count`, including punctuation and numbers.
  Build the table in a corpus pass, then reuse it in the feature pass. Unknown
  tokens receive zero. Record the corpus used to estimate frequencies.
- [x] For this exploratory corpus experiment, full-corpus TF is acceptable and
  must be declared. If held-out evaluation is added later, fit TF, vocabulary,
  and SVD only on the training portion.
- [x] Make raw lowercase word identity optional. It already exists in `postag/features.py`,
  but can encourage clusters of repeated words. Start with the requested affixes,
  length, case/digit flags, context, TF, and BOS/EOS; omit the CRF constant bias.
  Compare word identity as a separate variant rather than silently including it.
- [x] Keep punctuation in stored tokens and neighbor context. Define separate
  row-selection masks for all-token inspection and lexical POS evaluation, so
  punctuation-only clusters do not inflate conclusions about word POS.
  Record treatment of numbers and artifacts such as standalone BOM tokens.

**Acceptance:** feature rows align exactly with original token occurrences; the
updated feature tests pass, and a corpus sample exercises the loader,
TF computation, sentence boundaries, and row IDs together.

### 2. Produce and save numerical vectors

- [x] Fit a sparse `DictVectorizer` on the chosen feature dictionaries. Keep
  categorical values distinct from numerical and Boolean values. Save its
  vocabulary and fitted transformer.
- [x] Inspect the numerical scales of length, TF, and categorical indicators.
  Establish a documented baseline and compare a numeric-scaling variant if
  needed; avoid centering/densifying the full sparse matrix.
- [x] Apply `TruncatedSVD` with a fixed random seed, initially comparing 30 and
  50 components where matrix dimensions permit. Handle small inputs explicitly.
  Save the fitted reducer and report retained explained variance.
- [x] Save the sparse feature matrix, reduced dense vectors, aligned row metadata,
  and configuration. Record dimensions, feature options, corpus/sample identity,
  random seed, dependency versions, and any scaling/normalization choices.
- [x] Add a dedicated experiment dependency specification after verifying
  compatibility with the project's Python environment and existing NumPy pins.
  Choose and document one HDBSCAN implementation; do not assume similarly named
  implementations have interchangeable settings.

**Acceptance:** saved vectors can be reloaded and traced to their original tokens;
fitted transforms can be reused without rebuilding a different feature vocabulary.
No POS labels or dictionary analyses enter feature construction or reduction.

### 3. Test whether HDBSCAN finds useful structure

- [x] Begin with a bounded, reproducible sample spread across documents. Select
  target token occurrences while retaining their full original neighbor context;
  do not concatenate sampled tokens into artificial sentences.
- [x] Estimate memory/runtime from sparse matrix size, reduced matrix size, and
  the pilot clustering run before attempting the full corpus. Keep sample limits
  and selection seed configurable and save the exact selected row IDs.
- [x] Run HDBSCAN on reduced vectors using a documented distance metric and a
  small parameter sweep over `min_cluster_size` and `min_samples`. Record any
  post-SVD normalization as part of the configuration.
- [x] Save cluster labels aligned to row IDs, including the noise label `-1` and
  membership strengths if available from the selected implementation.
- [x] Report cluster count excluding noise, cluster-size distribution, noise
  fraction, assigned-token count, and runtime. Handle all-noise and single-cluster
  outcomes explicitly without presenting them as successful POS separation.
- [x] Export representative tokens and sentence contexts per cluster, with token
  counts and distinct forms. Inspect the proposed adjective and infinitive
  endings, mixed-POS clusters, repeated-word clusters, and noise examples.
- [x] Compare a small set of feature variants on identical rows: token-only
  features; token plus neighbors; token plus neighbors plus TF. Optionally compare
  the additional word-identity features. Keep other settings fixed for comparisons.

**Acceptance:** one repeatable command produces vectors, cluster assignments,
and a readable inspection report. If resource limits prevent full-corpus HDBSCAN,
deliver the sampled experiment and document the measured limitation. Weak clusters
alone are not proof that the features contain no POS information.

### 4. Add dictionary-based POS evaluation

- [x] Locate the intended `dict_uk` repository/release and inspect its actual data
  format, license, tag scheme, and build/export requirements. Record the source
  and revision. Obtain inflected surface forms with analyses, not only lemmas.
- [x] Build a lookup from surface form to a **set** of coarse POS tags. Document
  the mapping from source tags and a case/apostrophe lookup policy, preserving
  original token text. Do not arbitrarily choose the first analysis.
- [x] Use forms with a single distinct coarse POS as the initial scored subset.
  Multiple morphological analyses with the same coarse POS need not be excluded.
  Count ambiguous-POS forms, unmatched forms, and excluded nonlexical tokens
  separately; show dictionary coverage both by occurrences and distinct forms.
- [x] Calculate V-measure, ARI, and purity on eligible non-noise occurrences.
  Report eligible count, clustered count, and noise fraction alongside every
  score. If also scoring noise as one label, mark it as a separate secondary
  result; noise is not a discovered POS class.
- [x] Purity is the sum of each cluster's largest POS count divided by the
  number of scored occurrences. Pair it with cluster counts and other metrics:
  many tiny clusters can have high purity. V-measure and ARI compare partitions
  directly and do not require assigning a POS name to every cluster.
- [x] Add cluster-by-POS distributions and majority POS names for inspection.
  Report undefined/empty evaluations as unavailable with the reason, not zero.
- [x] Compare feature variants on the same eligible rows, disclose differing
  noise coverage, and include simple reference partitions such as one cluster
  for all eligible occurrences. Treat parameter selection on these scores as
  exploratory, not held-out validation.

**Acceptance:** a report connects vectors, clusters, dictionary coverage,
ambiguity, and scores. If the dictionary cannot yet be obtained or parsed,
retain the completed vector/clustering deliverables and explicitly mark external
POS evaluation as pending; manual inspection is not a replacement gold standard.

## Delivery and completion criteria

Implemented in this order: corpus-to-features integration → persisted numerical
vectors → bounded HDBSCAN experiment → dictionary evaluation → feature comparison.
Focused tests cover row alignment, reproducible sampling, transform persistence,
ambiguous dictionary forms, noise handling, and metrics on a hand-checkable example.

Saved vectors, fitted transforms, cluster assignments, inspection reports, and
dictionary-based feature comparisons are available under `runs/`.
See [RESULTS.md](RESULTS.md) for the measured corpus experiment, commands,
observed configurations, and limitations. No supervised model or contextual
POS disambiguator has been trained.
