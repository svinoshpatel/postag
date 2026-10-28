# Architecture and Technical Specifications

This document defines the technical requirements, experiment design, data contracts, and guardrails for the Ukrainian semi-supervised CRF POS tagger. See [AGENTS.md](../AGENTS.md) for repository operating rules and [PLAN.md](../PLAN.md) for implementation task state. The prescribed architecture does not imply that its components have already been implemented or verified.

## 1. Project Overview & Goals

### Domain & Task
This project implements a **Semi-Supervised Conditional Random Field (CRF) Part-of-Speech (POS) Tagger** for the **Ukrainian language**. The goal is to evaluate whether using a baseline CRF tagger (`CRF-Gate`) to pseudo-label unannotated Ukrainian literature text improves overall tagging accuracy when retraining a final tagger (`CRF-Final`) on an expanded training dataset.

### Datasets & Naming Conventions
The project uses two main data sources:

1. **Annotated Corpus (`UD_Ukrainian-IU`, `data/raw/gold/`):**
   * **`train_gold`**: The immutable, human-annotated training split used to train the baseline `CRF-Gate` model.
   * **`val_gold`**: The immutable, human-annotated development split used to select CRF hyperparameters and pseudo-label confidence thresholds. Its labels must not become training targets, and its contents must not contribute to learned vocabulary mappings or corpus-derived feature statistics. Scores used for selection are development results, not final performance estimates.
   * **`test_gold`**: The shared, immutable human-annotated benchmark set. Used strictly for final comparative evaluation of `CRF-Gate`, `CRF-Final`, and the frozen gold-only control.
2. **Unannotated Corpus (`data/raw/literature/`):**
   * Original, unannotated Ukrainian literary `.txt` files parsed during inference to extract candidate pseudo-labeled tokens/sequences. Folder membership determines input eligibility; no manually maintained JSON manifest is required when the corpus changes. Annotated UD `.conllu` files belong exclusively under `data/raw/gold/`.
3. **Augmented Training Corpus:**
   * **`train_extended`**: `train_gold` plus complete pseudo-labeled sentences that pass the confidence rule below, contamination checks, and configured sampling limits. The confidence threshold defaults to `0.90` and may be selected through the predefined validation search.

### Gold CoNLL-U Input Contract

* **Split identity and availability:** Configure explicit `.conllu` paths under `data/raw/gold/` for `train_gold`, `val_gold`, and `test_gold`. Require the three paths to resolve to distinct files; reject aliases to the same physical file. Record the UD release identifier and file checksums. Preserve official split membership and sentence order; do not reshuffle or repartition gold data. This contract specifies expected inputs rather than asserting that files are already present. A missing configured split is a runtime error; do not substitute another split or fabricate data.
* **Format and boundaries:** Stream UTF-8 input with strict decoding. Require ten tab-separated, nonempty fields on annotation rows, in CoNLL-U order: `ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`, `DEPS`, `MISC`. Use blank lines as sentence delimiters and require a terminating blank line after the final sentence. Comments must precede their sentence's annotation rows. Preserve original forms and metadata separately from normalized feature inputs. The reference format is [UD CoNLL-U](https://universaldependencies.org/format.html).
* **POS instances:** Include only positive integer-ID syntactic-word rows in model sequences. Require their IDs to run consecutively from `1` within each sentence. Exclude multiword-token range rows and decimal-ID empty nodes from features, labels, and token counts. Recognize these ID forms structurally; never parse decimal IDs as floating-point numbers. Retain surface reconstruction information separately for provenance and text-only auditing.
* **Source labels and forms:** Use `UPOS` as the source annotation. Each included word must have a nonempty `FORM` and a label from the fixed UD inventory: `ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`, `VERB`, `X`. Validate source labels before applying any optional target mapping. Missing or unknown source labels are errors; do not replace them with `X`. Preserve a literal underscore in `FORM`. The inventory follows the [UD POS specification](https://universaldependencies.org/u/pos/).
* **Loader output:** Emit one immutable sentence record containing its split, source path, sentence ordinal, original integer token IDs, original forms, aligned original UPOS labels, and aligned target labels. The target labels equal the original UPOS labels unless tag mapping is enabled below. Each included token must have exactly one original UPOS label and one target label. Retain `sent_id`, sentence text, and reconstruction metadata separately for provenance and auditing. Feature extraction receives surface forms without gold annotations; the contamination-audit interface exposes text and identifiers without labels.
* **Failure behavior:** Fail on malformed gold records, sentences with no integer-ID words, or a missing final sentence delimiter. Report the source path, line number, and reason; never silently skip examples or repair original files. These are POS-loader checks, not a claim of complete dependency-tree validation.

#### Gold parser implementation boundary (M3 task 1, 2026-10-28)

`crf.data.conllu.iter_gold_sentences()` emits immutable `GoldSentence` records
with original UPOS and target labels using the configuration's already-validated
mapping (identical to UPOS by default). These are parser outputs,
not normalized `SentenceRecord` artifacts. The shared mapping API below
verifies target-space consistency; `GoldSentence.to_record()` and
`iter_gold_records()` assemble validated, normalized records without publishing
artifacts. Only the resulting record exposes feature forms. Test-label parsing
requires `final_evaluation=True` before source access. Both parser and audit take
a configured `InputIdentity` and repository root, verify its current opaque-byte
checksum before reading, and check streamed bytes again at EOF. Consumers must
exhaust iterators before treating a source as successfully validated; earlier
sentences can be yielded before a later malformed sentence is discovered.

`iter_gold_audit()` has a distinct label-free return type. It interprets only
IDs, FORM, selected surface/document comments and spacing metadata; the seven
annotation columns are checked only for framing/nonemptiness, never parsed or
validated as annotations. It does not call the labeled parser. Labeled outputs
add complete original comments separately; audit outputs cannot expose them.

Strict tab/field and delimiter checks precede `conllu`'s structural ID/comment
helpers because its general parser permits missing fields and EOF delimiters.
Word IDs must be consecutive; range rows must precede their covered words,
span at least two words, and not overlap or extend beyond the sentence. Empty
nodes have structural, unique decimal IDs and an already observed word anchor
(or anchor zero); dependency-tree validation remains outside this POS loader.
LF and CRLF input are accepted, a file-start BOM and malformed UTF-8 are rejected, and
diagnostics identify source path and physical line. Extra blank lines are ignored;
a comments-only sentence and an unterminated final sentence are errors.

Surface metadata retains word/range/empty-node IDs and original forms, plus
`SpaceAfter`, `SpacesAfter`, and `SpacesBefore` values only. Reconstruction emits
MWT surface forms once in place of their component words and ignores empty
nodes. It uses escaped UD whitespace metadata when supplied, otherwise
`SpaceAfter=No` or a default inter-token space, omitting a default trailing space.
An explicit `# text = ...` value takes precedence and is marked original;
otherwise text is marked reconstructed. Gold-word source spans remain unavailable
(`null` when later converted to dataset records). Document/paragraph markers
persist across sentences; no inferred grouping from sent_id is introduced.
No raw-sentence size/token filters apply to gold. Later orchestration must apply
the existing memory monitor and publish-once artifact lifecycle.

### Optional Target Tag Mapping

* **Default and configuration:** Tag mapping is disabled by default (`tag_mapping: {}`), preserving the 17-tag UD target inventory. To consolidate service tags, configure all four source tags to the same target name, for example:

  ```yaml
  tag_mapping:
    ADP: FUNC
    CCONJ: FUNC
    SCONJ: FUNC
    PART: FUNC
  ```

* **Mapping semantics:** Apply the configured mapping to validated original UPOS labels in memory or in separate derived artifacts only. For the example above, `ADP`, `CCONJ`, `SCONJ`, and `PART` become `FUNC`; all other labels remain unchanged. Original gold files on disk and the original UPOS labels retained by the loader must remain untouched. Reject mappings that omit any of the four service tags, include other source tags, or assign different consolidated targets. The consolidated target name must be a nonempty string without whitespace and must not collide with an original UD tag.
* **Consistent target space:** Fix and record the mapping before model selection. Use it identically for `train_gold` training targets and `val_gold`/`test_gold` evaluation targets. Train both `CRF-Gate` and `CRF-Final` directly in the resulting target space; gate predictions, marginals, and pseudo-labels use that same space. Mapping changes require a new experiment, retraining both models, and regenerating pseudo-labels. Do not obtain mapped confidence scores by merely renaming labels or combining marginals from an unmapped gate.
* **Reporting and compatibility:** Record the mapping and resulting target inventory with configurations, derived datasets, models, and evaluation reports. Report per-tag and aggregate metrics in that inventory. The example yields 14 target classes and cannot distinguish the four original service tags. Compare gate and final metrics only under the same mapping and identify consolidated-task results separately from the original 17-tag UPOS results. `FUNC` is a custom target label, not a valid standard UD UPOS value; do not write mapped labels into original gold files or present mapped output as standard UD UPOS annotation.

#### Shared label-space implementation (M3 task 2, 2026-10-28)

`crf.data.labels.TagMapping` owns the fixed source inventory, service-tag rule,
immutable canonical mapping entries, source-label validation/application, target
validation and alphabetical target inventory. Configuration retains its existing
tuple representation and serialized `tag_mapping` object; `PipelineConfig.label_space`
and `PipelineConfig.target_inventory` expose the shared validated API. The existing
`crf.utils.config.UPOS` and `SERVICE_TAGS` imports remain compatible re-exports.
Invalid direct tuple construction (including duplicate sources) fails rather than
silently losing entries through `dict()`. Target names retain exact case and Unicode;
they must be valid UTF-8 strings without whitespace or NUL, and cannot be UD tags.

Configuration, labeled gold parsing and gold/pseudo dataset validation all use
this API. Gold labels are validated in the original UPOS space before mapping;
pseudo targets are validated directly in the resulting inventory and never remapped.
Neither original forms/labels nor source files change. Audit reading does not
access the label-space API or interpret labels. Label-space/configuration errors
remain `ConfigError` at configuration boundaries and `ValidationError` at dataset
boundaries; gold-row source-label failures retain source path/line diagnostics.

Mapping is preserved by existing effective-config serialization/hashing and target
arrays by the dataset schema; no wire-format/schema version change is required.
The inventory is derived from configuration rather than added as an unknown
configuration/dataset field. Future models, metrics and prediction stages must
consume this same API and persist/check their already-prescribed inventory fields.
This label-space task does not implement those stages; gold-record normalization
is supplied by the shared normalizer below. Mapped confidence aggregation or post-hoc prediction relabeling is not
provided.

### Core Methodology: Two-Stage Self-Training Workflow
1. **Stage 1 — Baseline (`CRF-Gate`):**
   * Fit the nine regularization candidates defined in the Training & Model-Selection Contract using `train_gold` only.
   * Evaluate candidates on `val_gold` and retain the selected fitted model as `CRF-Gate`.
   * Record baseline development metrics on `val_gold`. Defer evaluation on `test_gold` until Stage 4.
2. **Stage 2 — Pseudo-Labeling:**
   * Discover eligible `.txt` files directly under `data/raw/literature/`, validate their paths, and apply the Raw-Text Segmentation Contract before running `CRF-Gate` inference. Apply the source-isolation and corpus-contamination checks below before inference.
   * Decode each complete sentence using Viterbi prediction. Obtain token-label marginals using `predict_marginals`, and retrieve each predicted Viterbi label's marginal probability.
   * Retain a complete candidate sentence only when every predicted label's marginal is at least the lowest threshold in the frozen search grid. Preserve all tokens, labels, their order, and full-precision confidences. Stage 3 applies each configured threshold independently before sampling; do not discard candidates using a threshold that has not yet been selected on validation data.
3. **Stage 3 — Retraining (`CRF-Final`):**
   * For each configured confidence threshold, combine `train_gold` and the reproducibly selected, capped pseudo-label dataset into that threshold's `train_extended`.
   * Evaluate nine regularization candidates for each threshold, for 27 final-model configurations. Fit candidates from scratch when pseudo-labels are selected; for an empty selection, reuse the corresponding fitted gold-only baseline candidates under the sampling contract. Keep the selected gate fixed throughout.
   * Evaluate candidates on `val_gold`, jointly select the threshold and regularization settings, and retain the selected fitted model as `CRF-Final`.
4. **Stage 4 — Final Evaluation & Comparative Analysis:**
   * Retain the already-trained gold-only candidate with the same regularization settings as `CRF-Final` as `CRF-GoldControl`; no additional training is required.
   * Freeze `CRF-Gate`, `CRF-Final`, `CRF-GoldControl`, preprocessing, training-dataset identifiers and automatically recorded literature input paths, confidence threshold, and evaluation settings before accessing test results.
   * Evaluate all three models on the same `test_gold` instances in one final comparative evaluation under the Evaluation Contract.
   * Report `CRF-Final` versus `CRF-Gate` as the primary pipeline comparison and `CRF-Final` versus `CRF-GoldControl` as the comparison at matched regularization settings. Preserve negative differences and distinguish benchmark results from claims about literary-domain accuracy.

### Training & Model-Selection Contract

Use a first-order linear-chain CRF trained by regularized conditional log-likelihood through `sklearn-crfsuite`. Use `algorithm="lbfgs"`; CRFsuite internally uses OWL-QN when L1 regularization is nonzero. Train every candidate from scratch except for the explicitly permitted reuse of gold-only candidates when pseudo-label selection is empty. The reference settings are documented in the [CRFsuite API](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html).

| Fixed setting              |   Value |
|----------------------------|--------:|
| `max_iterations`           |     200 |
| `num_memories`             |       6 |
| `epsilon`                  |  `1e-5` |
| `period`                   |      10 |
| `delta`                    |  `1e-5` |
| `min_freq`                 |       0 |
| `all_possible_states`      | `False` |
| `all_possible_transitions` |  `True` |
| Pipeline seed              |      42 |

Enabling all label transitions permits transitions absent from the training examples; leaving all possible state features disabled limits feature expansion.

* **Execution and stopping:** Run candidate fits sequentially under the pipeline memory budget. Propagate the seed to applicable random operations; do not invent a `random_state` argument for the CRF estimator. Record effective parameters, dependency versions, termination reason, and iteration count. A model stopped at its iteration limit must be identified as budget-limited, not reported as converged.
* **Baseline search:** Fit the nine combinations of `c1` in `{0.01, 0.1, 1.0}` and `c2` in `{0.01, 0.1, 1.0}` using `train_gold` only. Select `CRF-Gate` by highest token accuracy on `val_gold`; break ties by higher macro-F1, then lexicographically smaller `(c1, c2)`. Retain that exact fitted model.
* **Threshold and final-model search:** Hold the selected gate fixed. Evaluate thresholds `tau` in `{0.90, 0.95, 0.98}`. For each threshold, construct a reproducible, capped pseudo-label dataset and evaluate the same nine regularization combinations on `train_extended`, fitting from scratch unless the selection is empty and baseline reuse applies. Select the final configuration by validation token accuracy, then macro-F1, fewer pseudo-labeled training tokens, higher threshold, and finally lexicographically smaller `(c1, c2)`. Keep validation data outside every fit. The default search comprises nine baseline configurations plus 27 final-model configurations, for 36 planned candidate configurations before any additional ablation experiments. Empty selections reduce actual training runs through explicitly recorded baseline reuse.
* **Failure and experiment boundaries:** Record every candidate configuration and result. Optimization errors, nonfinite objectives, or memory-budget failures leave the search incomplete; do not silently omit failed candidates and call the search complete. Freeze the search grid before development evaluation. Changing it creates a separately recorded experiment. Never refit selected models on validation data or use test results to revise selection.

### Raw-Text Segmentation Contract

* **Backend and reproducibility:** Use `tokenize_uk`'s non-legacy Ukrainian sentence and token span APIs through a reusable adapter in `src/crf/data/`. Pin the reviewed package version and its dependencies during environment setup and record the adapter version and segmentation configuration with each run. Use the same adapter for literature pseudo-labeling and raw-text inference. Gold sentences retain their supplied segmentation. The reference APIs are documented in [tokenize_uk](https://github.com/lang-uk/tokenize-uk/blob/master/docs/api.md).
* **Decoding and paragraph boundaries:** Decode incrementally using UTF-8 by default, permitting an explicitly configured source encoding. Decoding errors must fail with source diagnostics; do not silently replace characters. Blank lines delimit paragraphs; EOF closes the final paragraph. Single line breaks remain within a paragraph and must not independently force sentence boundaries. Never join text across files or paragraphs.
* **Sentence segmentation and alignment:** Apply the shared normalization contract before segmentation while retaining original text and alignment to its decoded character offsets. Segment complete paragraphs using Ukrainian sentence rules. Do not split sentences at read-buffer boundaries. Preserve source identity, paragraph/sentence ordinals, original token spans, and normalized feature forms. Discard whitespace-only results.
* **UD compatibility:** Apply versioned token-boundary rules consistent with [Ukrainian UD](https://universaldependencies.org/uk/): separate punctuation, compound hyphens, and abbreviation periods; preserve internal word apostrophes and digit groups such as `1 000 000`. Sentence splitting remains abbreviation-aware even when tokenization separates abbreviation periods. Do not invent multiword-token expansions. Initially, reject pseudo-label sentences containing known multiword-token surface forms observed in `train_gold`, unless a versioned expansion rule supports them. Report this exclusion and residual segmentation limitations. Check compatibility using reconstructed training text; do not tune segmentation against test data.

Resource limits are configurable and must be recorded with each run:

| Configuration                                 |                   Default |
|-----------------------------------------------|--------------------------:|
| Maximum buffered paragraph                    | 65,536 decoded characters |
| Maximum raw sentence                          | 16,384 decoded characters |
| Maximum raw sentence tokens, after adaptation |                       256 |
| Literature batch token limit                  |                     4,096 |

Character limits count decoded Unicode code points before normalization. The sentence token limit applies after token-boundary adaptation. If a paragraph exceeds its buffer limit, discard that whole paragraph while streaming to the next paragraph boundary. Reject oversized sentence candidates intact. Never turn a rejected paragraph or sentence into artificial training fragments. Report rejection counts and affected text volume. These filtering limits do not apply to gold evaluation sentences.

The tokenizer adapter's compatibility with the selected gold release must be verified when the gold files are available; package selection alone does not establish compatibility.

### Confidence & Sequence Acceptance

For a token sequence $\mathbf{x}$, let $\hat{\mathbf{y}}$ be its Viterbi-predicted label sequence and $c_i$ the marginal probability of that predicted label at position $i$:

$$
\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P_\theta(\mathbf{y}\mid\mathbf{x}),
\qquad c_i = P_\theta(Y_i = \hat{y}_i\mid\mathbf{x}).
$$

The confidence acceptance rule, with threshold $\tau$, is:

$$
\operatorname{accept}(\mathbf{x}) \iff |\mathbf{x}| > 0
\;\land\; \min_i c_i \ge \tau.
$$

The minimum marginal is a sentence acceptance score, not a joint sequence probability or a guarantee of empirical accuracy. Do not multiply token marginals or substitute each position's maximum marginal for the marginal of its Viterbi label. Compare unrounded probabilities. Empty sentences are ineligible; nonfinite or out-of-range probabilities must fail the run. Report acceptance rates by sentence length and predicted POS to expose selection bias.

### Pseudo-Label Sampling Contract

* **Token budget:** Let `N_gold` be the number of supervised tokens in `train_gold`. The effective cap is `max_pseudo_tokens = floor(pseudo_to_gold_ratio * N_gold)`, with ratio `1.0` by default. Count tokens after parsing/tokenization; tag mapping does not change token counts. Fix and record the ratio and seed before the model-selection search.
* **Reproducible ordering:** After contamination removal and deduplication, assign each sentence a priority by hashing the UTF-8 string `{seed}:{sentence_fingerprint}` with SHA-256. Represent the seed as a nonnegative decimal integer and the fingerprint as lowercase hexadecimal. Sort ascending by priority, breaking ties by sentence fingerprint. Use disk-backed sorting rather than loading all candidates into memory. The same sentence receives the same priority across all three thresholds.
* **Whole-sentence selection:** For each threshold, consider only sentences that meet its confidence rule. Walk them in priority order. Include a sentence if all its tokens fit within the remaining budget; otherwise skip it and continue. Never split a sentence, exceed the cap, or duplicate sentences to fill unused space. The cap may remain partly unfilled. Selected datasets at different thresholds need not be subsets of one another after sampling.
* **Training consistency:** Freeze one selected dataset per threshold and reuse it for all nine regularization candidates. Preserve the official gold sentence order, then append pseudo-labeled sentences in the priority order. Include each gold sentence and selected pseudo-labeled sentence once, without confidence weighting or automatic POS balancing. Record eligible/selected counts, budget exclusions, cap utilization, source distribution, and predicted-label distribution.
* **Empty selections:** Never lower the threshold automatically when no sentences are selected. Mark that threshold as a no-augmentation case. Reuse the corresponding nine cached gold-only candidates instead of repeating identical fits, recording the reuse explicitly. If every threshold selects nothing, report that self-training could not be performed and retain the selected baseline. A reused baseline result must not be presented as a newly trained augmented model or as evidence of augmentation gains.

### Evaluation Contract

The following metric definitions apply consistently to validation and final test evaluation. Test evaluation remains restricted to Stage 4.

* **Instances and alignment:** Evaluate on the supplied gold sentence boundaries and integer-ID words, without retokenizing gold text. Include punctuation, symbols, and `X` in primary metrics. Require matching sentence identities, token order, and prediction lengths. Missing, extra, or invalid predictions are errors; never silently truncate comparisons. An empty evaluation split is an error.
* **Accuracy and F1:** Token accuracy is the total correctly predicted tokens divided by all evaluated tokens, not the average of sentence accuracies. Use it as the primary model-selection metric. Report per-tag precision, recall, F1, and support. Macro-F1 is the unweighted mean over the fixed target inventory: all 17 UPOS labels, or all 14 labels under the approved `FUNC` mapping. Use `zero_division=0`, retaining absent classes in that mean. Micro-F1 may be reported but equals accuracy for this single-label, full-inventory evaluation. These averaging and zero-division definitions follow [scikit-learn metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).
* **Target inventory:** Evaluate predictions against original UPOS labels when mapping is disabled, or against identically mapped gold targets when enabled. Use the recorded target inventory consistently for per-tag metrics and averaging; do not mix original and consolidated labels in one comparison.
* **OOV diagnostics:** Define OOV tokens against the normalized, lowercased lexical forms in `train_gold` only. Use that same vocabulary and evaluation-token subset for every compared model, even when a model encountered those forms in literature. Report OOV token count and accuracy. If there are no OOV tokens, report accuracy as `null` with count zero.
* **Punctuation diagnostics:** Additionally report non-punctuation accuracy, excluding tokens whose original gold UPOS is `PUNCT`. Determine exclusions from gold labels, never predictions. For an empty diagnostic subset, report accuracy as `null` with count zero.
* **Reports and interpretation:** Store metric values on a 0–1 scale, plus token counts and a confusion matrix with gold labels as rows and predictions as columns in alphabetical target-label order. Report accuracy differences as `100 * (final_accuracy - gate_accuracy)` percentage points, preserving negative results. Gains on the UD benchmark do not establish accuracy on literary text without labeled literary evaluation.
* **Paired bootstrap:** For the final test comparison of `CRF-Final` and `CRF-Gate`, compute a paired bootstrap interval for the accuracy difference using 2,000 resamples and seed 42. Resample complete documents with replacement when document grouping is available and complete; otherwise resample sentences with replacement and disclose the fallback. Use the same sampled units for both models and report the 2.5th and 97.5th percentiles. Use cached predictions; do not retrain or revise selection. With fewer than two resampling units, report the interval as unavailable. The interval describes evaluation-sample uncertainty, not training variability. See the [paired bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
* **Gold-only control:** Retain the already-trained gold-only candidate with the same `c1` and `c2` as the selected final model, with the same feature configuration and target mapping. Freeze this control before test evaluation and report its results alongside the primary comparison. This requires no additional training and helps distinguish augmentation effects from differences in selected regularization.

---

## 2. Tech Stack & Architecture

### Technology Stack
* **Language & Runtime:** Standard CPython 3.14, managed through `uv` in a project-local environment. Declare `requires-python = ">=3.14,<3.15"` in `pyproject.toml`, select the interpreter through `.python-version`, and maintain `uv.lock`. Record the exact Python patch version and resolved dependencies with each run. Compatibility must be verified during environment setup before this runtime migration is declared complete.
* **Annotated Data Parsing:** `conllu` parser library (for CoNLL-U format handling).
* **Raw-Text Segmentation:** `tokenize_uk` through the versioned Ukrainian UD compatibility adapter defined above.
* **CRF Modeling Engine:** `sklearn-crfsuite` (wrapper over `python-crfsuite`).
* **Model Persistence:** `joblib` with the JSON metadata sidecars defined in the artifact contract.
* **Data Processing & Analytics:** `scikit-learn`, `numpy`, `pandas`.
* **Testing Framework:** `pytest`.

### Repository Structure & Architectural Blueprint
All code and assets must strictly adhere to the following directory layout:

```text
.
├── .python-version     # Selected CPython 3.14 interpreter
├── pyproject.toml      # Package metadata, dependencies, tooling, and src-layout packaging
├── uv.lock             # Reproducible dependency resolution
├── config/             # UTF-8 YAML pipeline configurations
├── data/
│   ├── raw/            # Immutable original datasets, separated by source role
│   │   ├── gold/       # Annotated UD_Ukrainian-IU .conllu splits
│   │   └── literature/ # Unannotated .txt files eligible for pseudo-labeling
│   ├── interim/        # Intermediate parsed tokens, features, or candidate pseudo-labels
│   └── processed/      # Final cleaned arrays/features ready for model training
├── docs/               # System architecture specs and methodology notes
├── models/             # Run-specific candidate bundles (.joblib / .json) and selected references
├── notebooks/          # Exploratory Jupyter notebooks (prefixed: 01_explore.ipynb)
├── reports/
│   └── <run_id>/       # Run-specific metadata, predictions, metrics, and bootstrap results
│       └── figures/    # Generated metrics plots and confusion matrices
├── scripts/            # Standalone CLI entry points importing crf
├── src/                # Main source directory; not itself the import package
│   └── crf/            # Reusable core package, imported as crf
│       ├── __init__.py
│       ├── data/       # Dataset loading, CoNLL-U parsing, text extraction
│       ├── features/   # Token feature engineering and contextual window extractors
│       ├── models/     # CRF wrappers, confidence scoring, pseudo-labeling logic
│       ├── utils/      # Logging, config loaders, path helpers
│       └── evaluation.py # Metrics calculations (accuracy, F1, confusion matrix)
└── tests/              # Pytest unit and integration tests for src/crf/

```

### Core Package and Migration Boundary

The CRF pipeline uses the `src/` layout with its core package at `src/crf/`, imported as `crf`. Configure the build backend in `pyproject.toml` to package and install this core into the project environment. Pipeline entry points under `scripts/` delegate to `crf`; they must not depend on importing `src` as a package.

The existing exploratory files under `postag/` are no longer required interfaces and may be moved or replaced during implementation. This migration establishes one core package that follows the contracts in this document rather than preserving incompatible legacy behavior. Update affected callers and tests with the migration; legacy feature representations and model artifacts are not interchangeable with the new schema without explicit verification. Existing files may remain while the migration is incomplete, but the new pipeline must use `crf`. The blueprint specifies the target layout, not a claim that migration has already occurred.

### Milestone 1 Migration Decision (2026-10-28)

The previous `postag` modules implement corpus-frequency features, a ±1 context window,
SVD/clustering, and whole-file spaCy tokenization. These conflict with the prescribed
feature schema and bounded preprocessing. Replace them with installable `crf`
foundations rather than retaining a second production interface. The sparse vector/SVD
experiment is retired because CRFsuite consumes feature dictionaries directly. Old
`postag` imports, feature keys, CLI switches, and exploratory serialized models are
not supported by the new package.

Milestone 1 establishes the surface-form feature extractor and a low-level tokenizer
backend operating on a caller-supplied paragraph. The backend preserves original slices
using upstream span offsets and uses the non-legacy APIs without spaCy. It is not yet
the production raw-text/UD adapter: integration of the shared aligned normalizer,
UD boundary adaptation, bounded file streaming, exclusions, and provenance remain
milestone 4 work.
The preview entry point accepts text only and writes JSON to stdout; it produces no
training artifacts and is not a substitute for the future pipeline or prediction CLIs.

## 3. Artifact & Run Contracts

### Run-Specific Locations

Generate a unique `run_id` automatically. Write each run into new directories; reject output paths that would overwrite existing artifacts or resolve inside `data/raw/`. Final evaluation adds new outputs to the same frozen experiment's run directories without overwriting development artifacts. Standalone prediction creates a fresh run directory under `reports/`.

| Location                                     | Contents                                                                                    |
|----------------------------------------------|---------------------------------------------------------------------------------------------|
| `data/interim/<run_id>/`                     | Pseudo-label candidates, contamination indexes, sampling intermediates                      |
| `data/processed/<run_id>/`                   | Derived gold records and one frozen augmented dataset per threshold                         |
| `models/<run_id>/candidates/<candidate_id>/` | `model.joblib` and `metadata.json` for each fitted candidate                                |
| `models/<run_id>/selected.json`              | Candidate IDs and checksums identifying gate, final, and control                            |
| `reports/<run_id>/`                          | Configuration, input identities, trial results, predictions, metrics, and bootstrap results |

Reused models reference their existing candidate IDs; do not create records claiming another fit occurred. Keep models on disk between fits rather than retaining all candidate estimators in memory.

### Run Lifecycle Foundation (2026-10-28)

`allocate_run()` generates UUID-based ASCII run IDs. Development reserves new
report, interim, processed, and model directories; prediction reserves only a
new report directory. Existing run paths in any artifact tree cause a new ID
to be tried, never reuse. A failure after partial allocation leaves incomplete
directories without completion metadata; there is no automatic resume or
rollback of existing artifacts. Allocation is intentionally nondeterministic
and does not affect seeded data ordering or model selection.

Artifact writers use directory-relative operations with symlink following
disabled on directory traversal. A fresh temporary file in the destination
directory is flushed/fsynced and validated before a hard-link operation
publishes the final name without replacement. Concurrent publishers can have
only one winner; existing files, directories, hard links and dangling symlinks
are preserved. Failed writes/validation never publish a final filename.
Only the writer's own temporary entry is cleaned up. Filesystem support for
these POSIX operations is required; unsupported operations fail explicitly.
This protects against ordinary concurrent writers/path redirection, not hostile
renaming of the repository itself or guaranteed durability after power loss.

`publish_completion()` requires a caller-supplied schema/integrity validator
and a memory monitor. It validates an encoded snapshot, checks the containing
run ID, status/error consistency and observed peak, and requires monitoring to
have finished. Successful publication additionally requires a successful monitor
context; a body exception prevents it even if no memory violation occurred.
`load_completed()` runs the same supplied schema/integrity validation and rejects missing, failed, malformed
or inconsistent completion metadata. It never treats directory existence as
success. These generic helpers do not replace later development/evaluation/
prediction validators: callers must validate all stage-specific fields, linked
checksums, models, trial counts and metric consistency before authorizing a
completed run. A failed descriptor may be published despite a latched resource
failure; it cannot authorize success.

`configure_logging()` configures the `crf` logger hierarchy on stderr with
propagation disabled and leaves unrelated application loggers alone. Repeated
configuration replaces its handler so messages are not duplicated; result data
remains the CLI's responsibility on stdout.

`MemoryMonitor` is a mutable runtime controller because samples, peak memory,
thread state and the first failure change during execution. Its context samples
at entry, in a background thread (default 0.1 seconds), at explicit checkpoints,
and on exit. Samples sum RSS bytes for the main process and recursively found
workers, counting each process once. Exited children may disappear between
samples; access-denied/root-monitoring errors fail closed. Budget equality is
permitted; a strict exceedance or monitoring failure is latched permanently.
Background failure interrupts the pipeline's main Python thread; checkpoints
and context exit raise the recorded resource exception. Callers must own their
pipeline process, use the context on its main thread, and clean up owned workers
in `finally` blocks when interruption unwinds execution. The foundation does not
kill unrelated processes or publish failure descriptors automatically.
Native code may delay Python interruption, short spikes may occur between
samples, and shared pages can be counted in multiple processes; this remains
best-effort aggregate RSS monitoring, not an operating-system memory limit.
Later training/CLI stages must use this context and guard completion publication.

### Streamable Dataset Records

Use UTF-8 JSONL: one complete sentence object per line, LF line endings, no BOM, and `schema_version: 1`. Reject unsupported schema versions and nonfinite numeric values. The reference format is [JSON Lines](https://jsonlines.org/).

Every dataset sentence object has exactly the following required fields, including explicit `null` values where specified:

| Field                 | Type and meaning                                                                                                                                     |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `schema_version`      | Integer `1`.                                                                                                                                         |
| `record_type`         | String `gold` or `pseudo`.                                                                                                                           |
| `sentence_id`         | Lowercase SHA-256 identifier constructed below.                                                                                                      |
| `source_path`         | Repository-relative POSIX path to the original input, without leading `/` or `.`/`..` path components.                                               |
| `source_sha256`       | Lowercase SHA-256 checksum of the original file bytes.                                                                                               |
| `source_role`         | String `train_gold`, `val_gold`, `test_gold`, or `literature`.                                                                                       |
| `sentence_ordinal`    | Positive integer counting sentences within the source before sentence-level filtering.                                                               |
| `paragraph_ordinal`   | Positive integer for literature; `null` for gold.                                                                                                    |
| `surface_text`        | Nonempty original sentence text, or reconstructed gold surface text.                                                                                 |
| `surface_text_origin` | String `original` or `reconstructed`.                                                                                                                |
| `token_ids`           | Array of positive integers: original consecutive gold word IDs, or literature IDs assigned consecutively from `1`.                                   |
| `original_forms`      | Array of nonempty original token-form strings.                                                                                                       |
| `normalized_forms`    | Array of nonempty, case-preserving token-form strings produced by the shared normalizer; lexical lowercasing remains a feature-extraction operation. |
| `token_spans`         | Array whose entries are `[start, end]` integer pairs or `null` when reliable source offsets are unavailable.                                         |
| `original_upos`       | Array of original UPOS strings for gold; `null` for pseudo-labels.                                                                                   |
| `target_labels`       | Array of mapped gold target strings or predicted pseudo-label strings in the configured target inventory.                                            |
| `pseudo`              | `null` for gold; an object with exactly the fields below for pseudo-labels.                                                                          |

The non-null `pseudo` object contains:

| Field                  | Type and meaning                                                                                       |
|------------------------|--------------------------------------------------------------------------------------------------------|
| `token_confidences`    | Array of finite numbers in `[0, 1]`, each the marginal of that token's Viterbi-predicted target label. |
| `min_confidence`       | Finite number equal to the minimum of `token_confidences`, without rounding.                           |
| `sentence_fingerprint` | Lowercase SHA-256 string from the Corpus Contamination Prevention contract.                            |
| `sampling_priority`    | Lowercase SHA-256 string from the Pseudo-Label Sampling Contract.                                      |
| `gate_model_sha256`    | Lowercase SHA-256 checksum of the gate's serialized model file.                                        |

Construct `sentence_id` by serializing the ordered array `[source_role, source_path, source_sha256, sentence_ordinal]` as JSON with `ensure_ascii=False`, `separators=(",", ":")`, and no trailing newline, then hashing its UTF-8 bytes with SHA-256. All SHA-256 fields contain exactly 64 lowercase hexadecimal characters. Sentence identity tracks source provenance; it is distinct from the normalized-text fingerprint used for overlap detection.

Gold sentence ordinals follow the supplied file order. Literature sentence ordinals count tokenizer sentence candidates across the file before sentence-level exclusions, so filtering leaves gaps rather than renumbering records. Paragraph ordinals count all nonempty paragraphs, including discarded oversized paragraphs. An oversized paragraph produces no sentence candidates; do not segment it merely to assign ordinals. Literature records use original surface text; gold text is original when supplied and reconstructed otherwise.

Spans use zero-based, end-exclusive Unicode code-point offsets into the original decoded source file before BOM removal, newline conversion, or character normalization. They are not byte offsets or offsets into normalized text. Non-null spans must satisfy `0 <= start < end <= decoded_source_length` and follow token order without overlap. Preserve alignment through normalization; use `null` for unavailable gold-word spans rather than inventing offsets from CoNLL-U annotation rows. Gold `token_ids` contain only integer-ID words, never multiword-token ranges or empty nodes.

All token-aligned arrays have the same nonzero length, including `original_upos` and `pseudo.token_confidences` when non-null. Validate original gold labels against the fixed UPOS inventory and require their mapped values to equal `target_labels`; pseudo-labels must belong to the recorded target inventory. Gold records require a gold source role and `pseudo: null`; pseudo-label records require `source_role: literature`, `original_upos: null`, and a non-null `pseudo` object. Preserve full-precision probabilities. Reject unknown fields, duplicate JSON keys, invalid field types, booleans in numeric fields, inconsistent identities/spans, and other schema violations with file/record diagnostics. Keep provenance and label fields outside feature dictionaries; retain additional original CoNLL-U metadata separately as required by the loader contract.

Create `test_gold` dataset records only during final evaluation. The earlier contamination audit uses its separate text-and-identifier interface without constructing records containing held-out labels. This dataset schema covers gold and pseudo-label training/evaluation inputs; standalone prediction outputs follow the separate prediction contract.

#### Dataset and identity validation boundary (2026-10-28)

Milestone 2 provides immutable `SentenceRecord`, `PseudoMetadata`, and
`InputIdentity` records through validated parsing factories. Dataset validation
requires the effective configuration and a matching input identity. If any token
span is non-null, callers must also provide the original decoded source length;
validation never guesses that length from sentence text or annotation rows.
Null spans remain valid when offsets are unavailable. Token IDs must equal
`1..N`; spans are ordered and disjoint across null entries too. Features receive
only the record's `feature_forms` tuple, never the record or its label/provenance
fields. Gold test records require an explicit final-evaluation opt-in; opaque
input hashing does not enable that opt-in or parse annotations.

These foundation validators enforce schema, label inventory/mapping consistency,
input bindings, sentence IDs, and seed-derived pseudo sampling priorities. They
validate fingerprint and gate checksums as SHA-256 strings; verification against
normalized source text and an actual model bundle belongs to the later
contamination/model-loading stages. Normalized forms are checked for type,
alignment and equality with the shared versioned normalization of original forms.
Original CoNLL-U comments/reconstruction metadata
remain the loader's separate provenance responsibility.

JSON readers reject duplicate keys at every depth, nonfinite values (including
overflowing exponents), BOMs, and invalid Unicode. JSONL reads one object at a
time, rejects blank lines and CRLF, and reports the path and physical line.
A complete final JSON object without a trailing LF is accepted on input;
encoded JSONL records always end in LF. Writers here return encoded bytes and
do not open or publish output files. Run allocation and atomic completion use the lifecycle foundation above; model/trial/metric schemas remain their later tasks.

Input identities are computed automatically in gold-role order followed by flat,
filename-sorted eligible literature files. Identity discovery validates source
paths and symlinks, excludes directories and non-`.txt` entries, rejects physical
gold aliases, and fails on an empty literature set. It reads original sources in
bounded binary chunks only, including test files as opaque bytes. Hashing checks
file metadata before/after reading and rejects observed changes; this is a
best-effort consistency check, not a filesystem snapshot. Content overlap and
known-source auditing remain milestone 4. Checksums describe exact original
bytes; an explicit verification helper checks a recorded identity against disk.

### Model Bundles

Serialize each successfully fitted estimator as `model.joblib` and store `metadata.json` beside it. The sidecar has exactly these required fields:

| Field                                             | Type and meaning                                                                                                          |
|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `schema_version`                                  | Integer `1`.                                                                                                              |
| `run_id`, `candidate_id`                          | Nonempty strings containing only ASCII letters, digits, `_`, and `-`. A candidate ID is unique within its run.            |
| `model_sha256`                                    | Lowercase SHA-256 checksum of the exact `model.joblib` bytes.                                                             |
| `training_dataset_sha256`                         | Lowercase SHA-256 checksum of the exact training JSONL bytes, including record order and line endings.                    |
| `training_sentence_count`, `training_token_count` | Positive integer totals for the training dataset.                                                                         |
| `pseudo_sentence_count`, `pseudo_token_count`     | Nonnegative integer pseudo-label counts included in those totals; both are zero for gold-only candidates.                 |
| `crf_parameters`                                  | Object recording the complete effective estimator parameters, including `algorithm`, `c1`, and `c2`.                      |
| `effective_config`                                | Complete resolved configuration conforming to the CLI & Configuration Contract.                                           |
| `target_inventory`                                | Array of unique target-label strings in alphabetical order, equal to the inventory implied by the configured tag mapping. |
| `python_version`, `unicode_version`               | Nonempty exact Python and Unicode database version strings.                                                               |
| `dependency_versions`                             | Object mapping installed dependency names to nonempty exact version strings.                                              |
| `seed`                                            | Nonnegative integer pipeline seed equal to `effective_config.seed`.                                                       |
| `termination_reason`                              | String `converged` or `iteration_limit`. The latter is budget-limited, not evidence of convergence.                       |
| `iterations`                                      | Nonnegative integer number of optimizer iterations performed.                                                             |

Record the parameters used for that candidate, not only the search grid or library defaults. Counts must agree with the ordered training records; pseudo-label counts cannot exceed their corresponding totals and must describe the same selected sentences. Training records remain under the run's processed-data location. Reused candidates retain their original metadata and training-dataset checksum rather than acquiring an augmented-training history they did not have.

The effective configuration carries the feature-schema version, normalization/tokenizer versions, tag mapping, encoding, and resource settings required for raw-text inference. Loading a standalone model bundle must not require access to the original training, validation, test, or literature files. Validate the metadata schema, model checksum, target inventory, preprocessing compatibility, and recorded dependency environment before inference; do not assume compatibility across library versions. Verify a save/load round trip during implementation. See the [model persistence reference](https://scikit-learn.org/stable/model_persistence.html).

Reject unknown or duplicate sidecar fields, invalid types, booleans in numeric fields, nonfinite numbers, and inconsistent metadata. SHA-256 values contain exactly 64 lowercase hexadecimal characters. The estimator-parameter and dependency-version objects are mappings with the meanings above, not permission to add unrelated top-level fields. Optimization errors, invalid objectives, and memory failures cannot produce a valid completed bundle.

### Frozen Model Selection

Write `models/<run_id>/selected.json` only after development completes successfully. It has exactly these required fields:

| Field                           | Type and meaning                                                                     |
|---------------------------------|--------------------------------------------------------------------------------------|
| `schema_version`                | Integer `1`.                                                                         |
| `run_id`                        | Identifier matching the containing run and referenced model metadata.                |
| `effective_config_sha256`       | Lowercase SHA-256 checksum of the resolved configuration encoded as specified below. |
| `selected_threshold`            | Finite number in `[0, 1]` belonging to the frozen threshold grid.                    |
| `gate`, `final`, `gold_control` | Objects containing exactly `candidate_id`, `model_sha256`, and `metadata_sha256`.    |

Each model reference identifies `models/<run_id>/candidates/<candidate_id>/`. Candidate identifiers follow the sidecar identifier rules; resolve paths within that run's candidate directory and reject escapes or metadata/checksum mismatches. `metadata_sha256` hashes the exact sidecar file bytes. Several roles may reference the same candidate; empty selections reuse baseline references without creating another fitted-model record. Require the gold-only control's `c1` and `c2` to match the final model's and its pseudo-label counts to be zero.

For `effective_config_sha256`, serialize the complete resolved configuration as JSON with `sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`, and `allow_nan=False`, without a trailing newline, then hash the UTF-8 bytes. Require every referenced bundle's effective configuration to produce that checksum. Fixed configuration-schema types govern serialization; validation must not silently coerce strings or booleans into numbers. This configuration checksum is distinct from checksums of arbitrary YAML source-file formatting.

Apply the same strict schema, type, finite-number, and checksum validation to frozen selection. Selection and model metadata are immutable after publication; final evaluation may read them but cannot replace them or revise the selected roles or threshold.

### Automatic Run and Prediction Records

Generate the following development-history files automatically under `reports/<run_id>/`. They are not manually maintained source manifests. All JSON/JSONL files use UTF-8 without a BOM, reject duplicate or unknown fields and nonfinite numbers, and enforce their declared types. Boolean values are not numeric counts or hyperparameters. Identifiers and checksums follow the model-bundle rules.

| File                    | Contents                                                                                                 |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| `effective_config.json` | Complete resolved configuration using the exact JSON encoding defined for `effective_config_sha256`.     |
| `inputs.jsonl`          | One source-identity object per gold split and eligible literature file.                                  |
| `trials.jsonl`          | One terminal record for every planned baseline/final configuration.                                      |
| `run.json`              | Immutable development-completion descriptor linking configuration, inputs, trials, and frozen selection. |

Every input object has exactly `schema_version` (integer `1`), `source_role`, `source_path`, `source_sha256`, `size_bytes` (nonnegative integer), and `encoding` (nonempty string). Roles and repository-relative paths follow the dataset schema. Record gold inputs in `train_gold`, `val_gold`, `test_gold` order, then literature files in deterministic discovery order. Gold encoding is `utf-8`; literature encoding is the effective-configured source encoding. Checksum and size describe original file bytes, not normalized text. Recording test-file identity does not authorize loading its labels during development.

Every trial object has exactly these fields:

| Field                                        | Type and meaning                                                                                                                                     |
|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `schema_version`, `run_id`, `trial_id`       | Integer `1` and run/trial identifier strings; trial IDs are unique within the run.                                                                   |
| `stage`                                      | String `baseline` or `final`.                                                                                                                        |
| `threshold`                                  | `null` for baseline; a finite configured threshold for final.                                                                                        |
| `crf_parameters`                             | Complete effective estimator parameters for that planned candidate.                                                                                  |
| `training_dataset_sha256`                    | Exact ordered training-JSONL checksum, or `null` if dataset construction did not finish.                                                             |
| `status`                                     | String `fitted`, `reused`, `failed`, or `not_executed`.                                                                                              |
| `model_ref`                                  | Object with exactly `candidate_id`, `model_sha256`, and `metadata_sha256`, or `null` when no successful candidate result exists.                     |
| `validation_accuracy`, `validation_macro_f1` | Finite numbers in `[0, 1]` for successful results; otherwise both `null`.                                                                            |
| `error`                                      | `null` for success; otherwise an object with exactly nonempty string fields `stage` and `message` describing failure or why execution did not occur. |

Successful `fitted` and `reused` trials require a valid model reference, dataset checksum, and both validation metrics. Reused trials reference the existing candidate and its cached validation results; the recorded dataset checksum must match that candidate's actual gold-only training history. Failed or unexecuted trials have `model_ref: null` and null metrics. A trial that failed after writing a model is still unsuccessful; an orphaned bundle is not a successful trial result. Preserve unavailable dataset checksums explicitly as `null` rather than inventing identities.

Record each planned configuration exactly once when it reaches a terminal outcome. At a handled abort, record remaining configurations as `not_executed`; do not silently shrink the planned grid. With the default grid, complete history has nine baseline and 27 final trial records, regardless of how many distinct fits were reused. Trials and fitted-model candidates have separate identities: several trial records may reference one candidate. A crash or truncated history remains incomplete and must never be interpreted as a smaller successful search.

The development descriptor `run.json` has exactly these fields:

| Field                               | Type and meaning                                                                                   |
|-------------------------------------|----------------------------------------------------------------------------------------------------|
| `schema_version`, `run_id`          | Integer `1` and the containing run's identifier.                                                   |
| `status`                            | String `complete` or `failed`, describing development only.                                        |
| `started_at`, `finished_at`         | UTC timestamp strings in ISO 8601 form ending in `Z`; finish must not precede start.               |
| `effective_config_sha256`           | Checksum of the configuration encoded under the frozen-selection rules.                            |
| `inputs_sha256`, `trials_sha256`    | Checksums of the exact corresponding JSONL file bytes; `null` if that file could not be completed. |
| `selected_sha256`                   | Checksum of the exact `models/<run_id>/selected.json` bytes on success; otherwise `null`.          |
| `source_revision`                   | Recorded Git commit identifier string, or `null` if unavailable.                                   |
| `source_dirty`                      | Boolean working-tree dirty-state observation, or `null` if unavailable.                            |
| `python_version`, `unicode_version` | Exact nonempty version strings for the execution environment.                                      |
| `dependency_versions`               | Dependency-name to exact-version mapping, using the model-bundle convention.                       |
| `peak_rss_mib`                      | Finite nonnegative observed peak aggregate resident memory in MiB.                                 |
| `error`                             | `null` for success; otherwise the same `stage`/`message` object shape used by trials.              |

Capture source/environment identity at run start and record the observed peak using the pipeline memory-monitoring contract. Seeds, preprocessing versions, and effective resource limits are available through the linked effective configuration. Record sampling statistics under the existing sampling and guardrail contracts; the completion descriptor does not replace those diagnostics.

Publish `run.json` once, after the development outcome and linked artifacts are finalized. A successful descriptor requires complete input/trial files, a successful fitted or permitted reused result for every planned configuration, and a valid frozen selection. Require its configuration/input/trial/selection checksums and candidate references to agree. A failed descriptor has `error` and cannot authorize evaluation. Missing completion metadata, partial files, invalid checksums, and absent trial results all mean an incomplete experiment; evaluation must refuse it. If an abrupt failure prevents a descriptor from being written, leave the run incomplete rather than fabricating a successful record.

Do not overwrite a descriptor or development-history artifact. Final evaluation adds its own completion record and outputs to the same run without changing the development history. These are terminal-history contracts, not authorization for automatic crash recovery or resuming partial artifacts.

### Prediction Records

Use sentence-level UTF-8 JSONL under the same strict JSON, schema-version, checksum, and type-validation rules as dataset records. Each prediction object has exactly these fields:

| Fields                                                            | Type and meaning                                                                                                                |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `schema_version`, `run_id`                                        | Integer `1` and the output run's identifier.                                                                                    |
| `model_ref`                                                       | Object containing exactly `candidate_id`, `model_sha256`, and `metadata_sha256`, identifying the generating bundle.             |
| `sentence_id`, `source_path`, `source_sha256`, `sentence_ordinal` | Source identity and positive sentence ordinal, following the dataset conventions and raw-input exception below.                 |
| `source_role`                                                     | String `val_gold`, `test_gold`, or `raw`.                                                                                       |
| `token_ids`, `original_forms`, `normalized_forms`, `token_spans`  | Nonempty, aligned arrays following the dataset definitions for IDs, case-preserving normalization, and original-source offsets. |
| `predicted_labels`                                                | Aligned array of labels in the generating bundle's target inventory.                                                            |
| `token_confidences`                                               | Aligned array of finite probabilities in `[0, 1]`, each the marginal of the actual Viterbi-predicted label.                     |

Prediction objects contain no gold target labels or original UPOS annotations. Require the same token identities, forms, order, and lengths as their evaluation inputs, without retokenizing gold. Preserve full-precision probabilities; reject missing, extra, duplicate, or misaligned sentence/token predictions. Test predictions remain restricted to final evaluation.

Standalone input uses `source_role: raw`. Record its repository-relative POSIX path when inside the repository and its resolved absolute POSIX path otherwise. Apply the sentence-ID formula with `raw` and that recorded path. This exception permits standalone inference on external text files without making them eligible for literature inputs for self-training. Use the recorded model's raw-text preprocessing and assign consecutive token IDs from `1`; preserve original-source token spans and sentence ordinals before sentence filtering.

Confidence acceptance, contamination exclusion, deduplication, known-multiword-token exclusion, and pseudo-label sampling caps apply to pseudo-label selection only. Standalone inference returns repeated and low-confidence sentences, preserving their predictions and confidences. Its filtering is limited to the shared decoding, segmentation, whitespace, and resource rules; no confidence threshold or training-corpus overlap filter may silently remove inference results. Known multiword-token compatibility limitations still apply and must be reported.

### Metric Records

Metric JSON has exactly `schema_version` (integer `1`), `run_id`, `split` (`val_gold` or `test_gold`), `target_inventory` (alphabetically ordered label array), and `models` (mapping described below). Each model entry has exactly these fields:

| Field                           | Type and meaning                                                                                                                                                                           |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_ref`                     | The generating candidate's three-field model reference.                                                                                                                                    |
| `sentence_count`, `token_count` | Positive integer evaluation counts.                                                                                                                                                        |
| `correct_token_count`           | Nonnegative integer no greater than `token_count`.                                                                                                                                         |
| `accuracy`, `macro_f1`          | Finite numbers in `[0, 1]` using the Evaluation Contract.                                                                                                                                  |
| `per_tag`                       | Mapping with exactly one entry per target label; each entry contains exactly `precision`, `recall`, `f1` (finite values in `[0, 1]`) and `support` (nonnegative integer gold token count). |
| `confusion_matrix`              | Square array of nonnegative integer counts, with gold rows and predicted columns in `target_inventory` order.                                                                              |
| `oov`, `non_punctuation`        | Objects containing exactly `token_count`, `correct_token_count`, and `accuracy`; counts are nonnegative integers with correct count no greater than token count.                           |

Require `accuracy = correct_token_count / token_count`, matrix totals equal token count, matrix trace equal correct count, row sums equal per-tag support, and macro-F1 equal the full-inventory mean of per-tag F1. Validate precision, recall, and F1 against the same confusion counts with the specified zero-division behavior. Diagnostic accuracy is the corresponding correct/total ratio, or `null` with both counts zero for an empty subset. All compared models use the same evaluation identities, target inventory, OOV mask, and gold-derived punctuation mask. Numeric comparisons must allow serialization-level floating-point error rather than comparing formatted display percentages.

Store each successful development trial's metric report at `reports/<run_id>/validation/<trial_id>/metrics.json`, with its trial ID as the sole `models` key. Its primary metrics must match the trial history. Reused trials use the cached baseline metric values and the existing model reference, without repeating inference or claiming another fit.

### Final Evaluation Reports

Create new outputs under `reports/<run_id>/evaluation/`:

| File                                              | Contents                                                                                                |
|---------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `gate.jsonl`, `final.jsonl`, `gold_control.jsonl` | Test prediction records for the three frozen roles, in identical gold sentence/token order.             |
| `metrics.json`                                    | Metric record with `split: test_gold` and exactly `gate`, `final`, and `gold_control` as `models` keys. |
| `comparison.json`                                 | Accuracy differences and primary paired-bootstrap result defined below.                                 |
| `units.jsonl`                                     | Ordered, label-free resampling-unit membership records defined below.                                   |
| `completion.json`                                 | Immutable evaluation outcome and integrity links defined below.                                         |

The comparison object has exactly `schema_version` (integer `1`), `run_id`, `split` (`test_gold`), `final_minus_gate_pp`, `final_minus_gold_control_pp`, and `bootstrap`. Both differences are finite signed percentage-point values calculated from unrounded accuracies. The bootstrap object has exactly `unit_type` (`document` or `sentence`), `unit_count` (nonnegative integer), `resamples` (positive integer), `seed` (nonnegative integer), `method` (`percentile`), `percentile_interpolation` (`linear`), `lower_pp`, `upper_pp`, and `unavailable_reason`.

Use the frozen bootstrap settings and the paired-resampling rules in the Evaluation Contract. For sorted replicate differences `d[0]` through `d[B-1]`, the percentile at probability `p` uses `h = (B - 1) * p` and linear interpolation between the floor and ceiling indices. Evaluate `p = 0.025` and `p = 0.975`; endpoints are finite percentage-point values with `lower_pp <= upper_pp`. On success, `unavailable_reason` is `null`. With fewer than two resampling units, both endpoints are `null` and `unavailable_reason` is a nonempty explanatory string. Sentence fallback must be disclosed through `unit_type` and the report diagnostics. Resampling concerns the final-versus-gate comparison only; the matched-control difference does not imply another bootstrap analysis.

Each `units.jsonl` object contains exactly `schema_version` (integer `1`), `unit_id` (the first member's sentence ID), and `sentence_ids` (a nonempty ordered array of sentence IDs). Use document groups only when preserved gold provenance assigns every test sentence to one complete, unambiguous document group; otherwise use one unit per sentence and disclose the fallback. Do not infer document identity from predicted labels or model outcomes. Order members by gold sentence order and units by first appearance. Require unique unit IDs and a disjoint, exhaustive partition of evaluated sentences. The number of records must equal `bootstrap.unit_count`.

Initialize NumPy `Generator(PCG64(seed))` with the frozen bootstrap seed. For each replicate, in sequence, draw `unit_count` independent uniform integer indices in `[0, unit_count)` with replacement using `integers(0, unit_count, size=unit_count, dtype=int64)`. Use those same indices for both models, retaining repeated units with their multiplicity. Recompute each model's pooled token accuracy from the sampled units' correct-token and token counts, then calculate the signed percentage-point difference. Do not average unit accuracies. With fewer than two units, record the unavailable interval without drawing replicates. Process replicates incrementally under the memory budget.

Evaluation completion has exactly `schema_version` (integer `1`), `run_id`, `status` (`complete` or `failed`), `started_at`, `finished_at`, `development_sha256`, `selected_sha256`, `outputs`, `environment`, `peak_rss_mib`, and `error`. Timestamp, peak-memory, and error definitions follow the development descriptor. `development_sha256` identifies the exact development `run.json`; require it and the selected-file checksum to match a valid frozen experiment. The `outputs` mapping associates filenames from the six prediction/metric/comparison/unit files above with their exact byte checksums. A complete result requires all six entries; a failed result may identify only files that were fully written and validated.

The `environment` object contains exactly `python_version`, `unicode_version`, `dependency_versions`, `source_revision`, and `source_dirty`, following development-record types. Record the actual evaluation environment and observe peak memory. Validate compatibility against frozen model metadata before prediction. Do not replace development environment observations with evaluation observations.

Publish completion only after the outcome and linked outputs are finalized and validated. A successful record has `error: null`; a failed record has an error object and cannot authorize cached success. Return cached final results only when successful completion, linked frozen history, model references, output checksums, alignments, and metric consistency all validate. Missing, partial, or failed evaluation remains incomplete; never overwrite it or rerun it automatically under the same completed-result identity.

### Standalone Prediction Reports

Write `predictions.jsonl` and `completion.json` into a fresh `reports/<run_id>/` directory. Completion has exactly `schema_version` (integer `1`), `run_id`, `status`, `started_at`, `finished_at`, `model_ref`, `input`, `predictions_sha256`, `sentence_count`, `token_count`, `environment`, `peak_rss_mib`, and `error`. Status, timestamp, environment, memory, error, and model-reference rules follow the contracts above. The `input` object contains exactly `source_role: raw`, `source_path`, `source_sha256`, `size_bytes`, and `encoding`, using the standalone path convention and recorded model encoding.

Counts are nonnegative integers and must match the output records. A successful result has a valid predictions checksum and `error: null`; a failure has an error object and a null predictions checksum if no complete validated file exists. An input yielding no eligible sentences may produce an empty JSONL file with zero counts; report filtering counts without inventing fragments or labels. Standalone reporting computes no gold metrics and does not modify model metadata or frozen experiments. Apply the same publish-once and incomplete-output rules.

All run metadata is generated automatically; changing the literature corpus does not require a manually maintained manifest.

## 4. Environment & Verification Contract

Declare direct runtime dependencies and development tools in `pyproject.toml`; reproduce the environment with `uv.lock` in the project-local environment. Include Ruff, mypy, and pytest as project development tools. Keep Ruff configuration in `pyproject.toml`, target Python 3.14, and document material lint-rule choices. See the [uv project structure reference](https://docs.astral.sh/uv/concepts/projects/layout/).

Milestone 1 established CPython 3.14.6 on macOS arm64 with `tokenize-uk==2.0.0`
(without its spaCy extra), `python-crfsuite==0.9.12`, and `sklearn-crfsuite==0.5.0`.
The complete resolved environment is in `uv.lock`; reproduce it with
`uv sync --locked`. Dependency imports, a tiny CRF fit/marginal/persistence
round trip, and an isolated installed-package import establish compatibility
for these foundations; they do not establish real-corpus or cross-platform results.

Ruff enables essential syntax/error checks (`E4`, `E7`, `E9`, `F`), import ordering
(`I`), Python modernization (`UP`), and common correctness pitfalls (`B`).
Formatting remains a separate check. Mypy uses strict mode for `src/crf/`.
Pytest uses importlib mode and the installed package, with no `sys.path` injection;
integration tests carry an explicit marker and document their dependencies.
`src/crf/py.typed` marks the installed package as typed. The Hatch build backend
packages only `src/crf/`.

The selected runtime is a target until the environment has been established and compatibility verified. Do not claim the existing environment has been upgraded merely because this specification has changed. Seeds control applicable random operations but do not guarantee identical floating-point results across platforms or dependency versions; record the effective environment and any observed nondeterminism.

Before producing artifacts, configure `.gitignore` to exclude generated datasets, model bundles, and run outputs. Original datasets remain immutable and generated artifacts must not be staged or committed, except for intentional small test fixtures permitted by the repository operating rules.

Run the following separate checks from the repository root when implementing executable behavior or configuration:

| Check      | Command                        |
|------------|--------------------------------|
| Lint       | `uv run ruff check .`          |
| Formatting | `uv run ruff format --check .` |
| Types      | `uv run mypy src/crf/`         |
| Tests      | `uv run pytest tests/ -v`      |

Record each check's actual result. Runtime compatibility and the package migration are not complete until applicable verification has been performed. Documentation-only changes are verified by inspection under `AGENTS.md` and do not establish that these executable checks have passed.

### PyCharm Integration Decision (2026-10-28)

PyCharm uses the existing project-local environment as a native uv SDK; it must
not create an independent environment or change dependency authority. During
setup, the enabled pyproject module model regenerated module roots relative to
`.idea` and discarded explicit roots/exclusions. Disable `usePyprojectToml` for
that IDE module model and maintain the explicit repository/source/test roots.
This affects IDE module generation only; `pyproject.toml`, `.python-version`,
and `uv.lock` remain authoritative for packaging, runtime and tooling. The
module inherits the project uv interpreter. See [PyCharm setup](PYCHARM.md).

### CLI & Configuration Contract

Entry points use `argparse`, run from the repository root, and delegate reusable logic to `crf` under `src/crf/`. The following are target commands to implement, not claims that the scripts already exist:

| Command                                                                                 | Behavior                                                                                                                                                           |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `uv run python scripts/run_pipeline.py develop --config config/pipeline.yaml --seed 42` | Create a new experiment, run Stages 1–3 under the approved search, and freeze the selected models and effective configuration after successful development.        |
| `uv run python scripts/run_pipeline.py evaluate --run-id <id> --seed 42`                | Load a successfully frozen experiment and perform Stage 4 without training or selection changes.                                                                   |
| `uv run python scripts/predict.py --model <bundle> --input <text.txt>`                  | Load one candidate bundle directory, apply its recorded raw-text preprocessing and target mapping, and write predictions to a fresh `reports/<run_id>/` directory. |

Development may access held-out test surface text only through the separate contamination-audit interface. Input-identity recording may read original test-file bytes to calculate checksums and sizes; this operation treats all content as opaque bytes and exposes no annotation values. The text-only audit may decode the file and use IDs, forms, surface-text comments, and spacing metadata for reconstruction, but must leave `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`, and `DEPS` uninterpreted. Its interface exposes only permitted surface text, reconstruction metadata, and identifiers. Neither path may construct or expose held-out target labels to model code. Development must not load test targets into training/evaluation records, generate test predictions, or compute test metrics. General configuration validation does not authorize parsing held-out labels. Test targets, predictions, and metrics are restricted to final evaluation. Standalone prediction is not an alternative development-time test-evaluation route.

Use one UTF-8 YAML configuration format with a mapping at the root. Load plain configuration values without executable object construction. Reject duplicate keys at every nesting level. Version 1 recognizes the following keys; nested objects accept only the listed children:

| Key or section   | Children and defaults                                                                                                                                                                                                                                                                                                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config_version` | Required integer `1`.                                                                                                                                                                                                                                                                                                                                                                      |
| `seed`           | Nonnegative integer; default `42`.                                                                                                                                                                                                                                                                                                                                                         |
| `tag_mapping`    | Default `{}`; otherwise the complete mapping defined in Optional Target Tag Mapping.                                                                                                                                                                                                                                                                                                       |
| `data`           | Required `train_gold`, `val_gold`, `test_gold`, and `ud_release`; `literature_dir` defaults to `data/raw/literature/`; `source_encoding` defaults to `utf-8` and applies to literature only.                                                                                                                                                                                               |
| `preprocessing`  | `normalizer_version` and `tokenizer_adapter_version`; each is an integer defaulting to `1`, identifying the initial contracts in this document. Reject unsupported versions.                                                                                                                                                                                                               |
| `features`       | `schema_version`; integer default `1` for the initial Feature Dictionary Contract. Reject unsupported versions; version selection does not authorize additional features or a different context window.                                                                                                                                                                                    |
| `training`       | `c1_grid`, `c2_grid`, and `parameters`; grids default to `[0.01, 0.1, 1.0]`. The `parameters` mapping accepts exactly `max_iterations`, `num_memories`, `epsilon`, `period`, `delta`, `min_freq`, `all_possible_states`, and `all_possible_transitions`, with their documented defaults. The algorithm is `lbfgs`; the pipeline seed remains top-level `seed`, not an estimator parameter. |
| `self_training`  | `thresholds` defaults to `[0.90, 0.95, 0.98]`; `pseudo_to_gold_ratio` defaults to `1.0`.                                                                                                                                                                                                                                                                                                   |
| `resources`      | `read_buffer_bytes: 65536`, `max_paragraph_chars: 65536`, `max_sentence_chars: 16384`, `max_sentence_tokens: 256`, `batch_tokens: 4096`, and `memory_budget_mib: 8192`.                                                                                                                                                                                                                    |
| `evaluation`     | `bootstrap_resamples: 2000` and `bootstrap_seed: 42`. Metric definitions and resampling-unit selection follow the Evaluation Contract.                                                                                                                                                                                                                                                     |

Required data paths and the release identifier have no inferred substitutes. Optional sections and values receive the documented defaults. Resolve relative paths, including CLI paths, against the repository root rather than the configuration file's directory. Require the three gold split paths to resolve to distinct physical files. Preserve the gold and literature folder restrictions and all resolved-path/output checks defined elsewhere in this document.

Reject unknown keys, incorrect types, nonfinite numeric values, empty required strings or grids, and duplicate grid/threshold entries. Booleans are not valid integers or numeric hyperparameters. Require positive integer resource limits, `batch_tokens >= max_sentence_tokens`, nonnegative finite regularization values and sampling ratio, and finite thresholds in `[0, 1]`. Validate CRF stopping parameters against their estimator-supported domains and require a positive integer bootstrap resample count and nonnegative integer seeds. The approved default experiment remains 36 candidate configurations with the documented fixed training/evaluation settings; any intentional change must be declared before selection and recorded as a separate experiment under the existing experiment-boundary rules.

For development, apply precedence in this order: documented defaults, YAML values, then an explicitly supplied `--seed`. An omitted CLI seed does not overwrite the configured seed. Record the complete effective configuration automatically. The pipeline seed controls applicable training/sampling operations; the bootstrap uses the separately recorded `evaluation.bootstrap_seed`.

#### Configuration validation implementation decisions (2026-10-28)

`crf.utils.config.load_config(path, repository_root=..., seed=None)` resolves the
configuration path against the explicitly supplied repository root. Immutable
records retain repository-relative `Path` values for corpus locations;
`to_dict()` emits all effective fields with POSIX path strings and numeric grids
as arrays. Numeric-valued settings accept integers or floats (never booleans)
and normalize to floats; integer-valued settings require integers. YAML merge
keys, recursive aliases, non-string mapping keys, and non-plain values are
rejected along with duplicates and unknown fields. Harmless nonrecursive aliases
are allowed. Only schema/preprocessing/feature version 1 is currently supported;
this does not assert that later preprocessing milestones are implemented.

CRF domain interpretation: `max_iterations` is nonnegative (zero means no
iteration limit in CRFsuite), `num_memories` and `period` are positive integers,
`epsilon`, `delta`, and `min_freq` are finite nonnegative numbers, and feature
expansion switches are booleans. Explicit overrides define a separate experiment;
the defaults still describe the approved 36 configurations. Source encodings
must resolve to a Python text codec supporting incremental decoding.

Configuration loading checks gold file existence, `.conllu` suffix, containment
under the repository's `data/raw/gold/`, and distinct device/inode identities
(including hard links). The literature directory must be the prescribed
`data/raw/literature/`, must exist, and must not traverse symlinks. Gold and
literature roots must resolve inside the repository and remain disjoint.
Eligibility discovery and content checks remain milestone 4. No labels are read.
`parse_config()` validates settings and path syntax without requiring input files;
it supports standalone model metadata whose original corpora may be absent.
Development must use `load_config()` or explicitly call `validate_input_paths()`.

`resolve_cli_path()` provides repository-relative CLI resolution while allowing
external standalone inputs. `validate_output_path()` checks a proposed fresh
output under `data/interim/`, `data/processed/`, `models/`, or `reports/`, rejecting
existing entries (including dangling symlinks), symlinked output ancestors,
resolved escapes, and raw-input locations. It performs no allocation; the lifecycle foundation provides race-safe
unique allocation and publication.

Freeze effective configuration, input identities, dataset identities, preprocessing versions, selected model references, and evaluation settings only after every planned configuration has a successful fitted or permitted reused result. Evaluation accepts no new configuration or parameter overrides. Its `--seed`, when supplied, is an assertion that must equal the frozen bootstrap seed; when omitted, use that frozen seed. Reject an incomplete experiment, changed inputs, incompatible preprocessing/dependencies, or a model/checksum mismatch before computing test predictions. If evaluation already completed, verify and return the stored results without recomputing them. Never overwrite partial or completed evaluation artifacts or present a failed attempt as complete.

Changing literature inputs is supported by starting a new development run, with fresh automatically generated input metadata; no manual source manifest is required. Diagnostics go to stderr and machine-readable result summaries go to stdout.

## 5. Project Constraints & NLP Guardrails

### Ukrainian Language & NLP Feature Constraints

1. **Character Normalization & Unicode Standards:**
* **Shared normalization and preservation:** Use one versioned normalizer for gold feature forms, literature preprocessing, raw inference, and contamination checks. Transform working copies only; preserve original files, forms, and alignment to decoded source offsets. Record the normalizer version and Python Unicode database version with each run.
* **Transformation order:** For raw text, remove an optional BOM (`U+FEFF`) only at file start and normalize CRLF/CR line endings to LF. Apply Unicode NFC normalization, then the apostrophe mappings below. Map nonbreaking space (`U+00A0`) and narrow nonbreaking space (`U+202F`) to ordinary space (`U+0020`). Preserve line breaks for paragraph segmentation. Apply the same character normalization to individual gold token forms without changing gold token boundaries. NFC provides canonical normalization; it does not replace the explicit apostrophe mapping. See [Unicode normalization](https://www.unicode.org/reports/tr15/).

| Input apostrophe                       | Normalized output |
|----------------------------------------|-------------------|
| `U+0027` — apostrophe                  | `U+0027`          |
| `U+2019` — right single quotation mark | `U+0027`          |
| `U+0060` — grave accent                | `U+0027`          |
| `U+02BC` — modifier letter apostrophe  | `U+0027`          |

* **Cyrillic preservation:** Preserve Ukrainian letter distinctions, including `г/ґ`, `и/і`, `е/є`, and `і/ї`, and preserve the soft sign. Do not transliterate or replace Latin/Cyrillic lookalikes. Do not silently strip combining stress marks, soft hyphens, or zero-width characters; retain them and report occurrences of formatting characters.
* **Casing before lexical lowercasing:** Compute `is_title`, `is_uppercase`, and `is_lowercase` on the normalized, case-preserving token using Python's Unicode-aware `istitle()`, `isupper()`, and `islower()`. These flags are independent. Then derive the lexical feature form by lowercasing and applying NFC again. Do not lowercase text before sentence segmentation. For example, `ҐАНОК` keeps its uppercase flag while its lexical feature form becomes `ґанок`. See [Python string operations](https://docs.python.org/3/library/stdtypes.html).

#### Versioned normalizer and source alignment (M3 task 3, 2026-10-28)

`crf.data.normalization` implements version 1 with the running Python Unicode
database version. `normalize_text()` accepts a complete decoded paragraph (or
bounded text unit) and its nonnegative absolute `source_offset`. It removes one
leading BOM only at absolute file offset zero, converts CRLF/CR to LF, then
applies NFC and the specified apostrophe/space mappings. It neither lowercases
nor collapses whitespace. Callers retain paragraph boundaries and must not split
CRLF pairs or Unicode normalization units at read-buffer boundaries; the future
M4 streaming adapter owns assembling complete paragraphs before normalization.
The existing low-level tokenizer/preview remains unchanged until that adapter
is implemented; it is not a production normalization/inference interface.

The immutable result retains original/normalized text, absolute source intervals
for each normalized character, formatting-character observations, normalizer
version and Unicode version. NFC normalization units include starters and their
combining marks and any interacting zero-combining-class compositions (including
Hangul). Every output character in a unit refers conservatively to the unit's
whole original interval, handling composition, expansion and canonical reordering
without claiming a one-to-one mapping. `source_span(start, end)` projects a
nonempty normalized interval to an original zero-based, end-exclusive code-point
span; it rejects out-of-bounds indices and cuts inside a normalization unit. This
prevents adjacent projected tokens from overlapping or dropping combining marks.
CRLF maps to both original characters. Removed BOMs are outside projected text.

`normalize_form()` applies the same character rules to a nonempty gold/token
form, preserving a leading BOM because a form is not a file; gold boundaries and
original strings remain intact. `normalize_forms()` preserves sequence order and
length. The old `crf.features.surface.normalize_form` import remains a compatible
re-export. Feature casing uses these case-preserving forms; `lexical_form()` then
lowercases a normalized form and applies NFC again, without case folding.

Unicode `Cf` formatting characters (including soft hyphens, zero-width characters
and embedded BOMs) remain present except the permitted file-start BOM removal.
Text results record every occurrence with its original offset and removal flag;
raw-text and form normalization emit a summary through module logging when such
characters occur. Combining stress marks remain present under NFC. Empty raw
text is valid; empty token forms, invalid Unicode scalar text, boolean/noninteger/
negative offsets and unsupported version values fail explicitly. No truncation, resource
filter, filesystem operation or corpus statistics are introduced here.

Gold conversion revalidates source/target/configuration consistency and explicit
test-label authorization through `parse_sentence_record()`, retains unavailable
gold-word spans as null, and leaves original comments/reconstruction metadata on
the parser output. Dataset validation rejects normalized forms inconsistent with
original forms for both gold and pseudo records. Wire formats remain version 1;
previously accepted incorrect normalization now fails. Later contamination,
literature and inference stages must use this normalizer and preserve its offsets;
this task does not claim full tokenizer compatibility or corpus readiness.

2. **Feature Dictionary Contract:**

The extractor accepts an ordered sentence of surface forms and returns one feature dictionary per token, preserving order and length. Training targets remain a separate, equally sized sequence. Empty sentences and empty token forms are errors. Dictionary keys are strings; values are strings, booleans, or finite floating-point numbers. Missing optional features are omitted, never represented by `None`.

Current-token features are:

| Key                                        | Type    | Meaning                                                          |
|--------------------------------------------|---------|------------------------------------------------------------------|
| `bias`                                     | float   | Always `1.0`                                                     |
| `word.lower`                               | string  | Lexical form from the normalization contract                     |
| `prefix.1` through `prefix.4`              | string  | First 1–4 characters of the lexical form                         |
| `suffix.1` through `suffix.5`              | string  | Last 1–5 characters of the lexical form                          |
| `is_title`, `is_uppercase`, `is_lowercase` | boolean | Case-preserving definitions from the normalization contract      |
| `has_hyphen`                               | boolean | Contains `U+002D`, `U+2010`, or `U+2011`                         |
| `has_digits`                               | boolean | Any character satisfies Python's `isdigit()`                     |
| `is_punct`                                 | boolean | Every character belongs to a Unicode punctuation category (`P*`) |

Affix lengths count Unicode code points after lexical normalization. Emit an affix only when the token has at least that many code points; do not pad or repeat shortened slices. For example, `гідність` yields `suffix.5 = "ність"`. Affixes are surface heuristics, not morphological analyses or agreement constraints.

* **Context and boundaries:** For offsets `-2`, `-1`, `+1`, and `+2`, include the neighboring token's `word.lower` and six shape flags, prefixed with the offset, for example `-1:word.lower` and `+1:is_uppercase`. Do not copy neighboring affixes or bias terms. If a neighbor lies outside the sentence, emit only its boundary flag, for example `-1:BOS = True` or `+2:EOS = True`. Never cross sentence boundaries.
* **Versioning and vocabulary behavior:** Use the same recorded feature-schema version and extraction configuration for both models and every pipeline stage. The initial schema contains no learned vocabulary mapping or IDF features. Emit unseen lexical strings normally; do not fit preprocessing on validation or test data. Reject unsupported keys, value types, and nonfinite numeric values.

### Strict Test Data Isolation & Leakage Prevention

1. **Immutable Test Set:**
* Except for the text-only contamination audit defined below, `test_gold` must never contribute to training, hyperparameter tuning, feature dictionary building, or pseudo-label candidate selection. The audit may use held-out surface text solely to exclude overlapping literature; it must not use held-out labels or model scores.
* **Opaque input identity:** Reading original test-file bytes solely for automatic checksum/size recording is permitted under the CLI & Configuration Contract. It does not authorize annotation parsing, feature fitting, predictions, or metrics before final evaluation.
* **Pseudo-label source isolation:** Inspect only immediate children of `data/raw/literature/`, in sorted filename order, and select regular files with the exact `.txt` suffix. Ignore other extensions and directories without traversing them. Pseudo-label discovery must not scan `data/raw/` or `data/raw/gold/` recursively or fall back to either directory as an input source. Gold files may be accessed by the separate contamination-audit component through explicitly configured split paths. Pseudo-label input discovery and inference must not read gold files.
* **Source path checks:** Before inference, reject a missing literature directory, an empty eligible file set, a symlinked literature directory or `.txt` candidate, and any candidate whose resolved path lies outside the resolved literature directory or inside the resolved gold directory. Annotated datasets and text exported from gold splits are ineligible regardless of filename. Record the selected repository-relative input paths automatically in run metadata. Folder separation does not establish content disjointness; separate corpus-contamination checks are required.

2. **Feature Leakage Prevention:**
* Vocabulary mappings and corpus-derived feature statistics, including document frequencies or IDF if used, must be fitted only on the applicable training dataset and any explicitly declared unannotated fitting corpus. Neither `val_gold` nor `test_gold` may contribute to these estimates. Confidence thresholds are selection hyperparameters and may be tuned on `val_gold` using a predefined, recorded candidate search. Neither `test_gold` inputs nor labels may be used for threshold tuning.
* **Allowed feature inputs:** Feature extraction must use surface token forms, text-derived context and sentence-boundary information, and preprocessing statistics fitted under the training-only rules above. Apply the same feature extractor during training, validation, pseudo-labeling, and test inference.
* **Annotation exclusion:** Training targets must remain separate from feature dictionaries. Gold `UPOS`, neighboring gold labels, `LEMMA`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`, and `DEPS` must never become predictive features. CoNLL-U IDs, comments, `MISC`, and source metadata may support parsing, alignment, or provenance but must not become predictive features.
* Test evaluation occurs only in Stage 4 after all model-selection decisions are frozen. Test results must not guide subsequent training, threshold selection, feature changes, or selection of the reported configuration. Benchmark logging records this final evaluation and does not authorize additional development-time test evaluations.

3. **Corpus Contamination Prevention:**
* Before fitting statistics from literature or generating pseudo-labels, compare literature sentences against all gold splits using a predefined, versioned surface-text normalization and exact-match fingerprints. Exclude matching literature sentences, preserve gold data unchanged, and deduplicate remaining literature sentences in deterministic discovery order. This prevents gold-training examples from being counted again as pseudo-labels and excludes exact validation/test overlaps.
* **Fingerprint construction:** Compute SHA-256 over the UTF-8 encoding of normalized surface sentence text after collapsing Unicode whitespace runs to one ordinary space and trimming surrounding whitespace. Preserve case and punctuation. Use gold sentence text when available; otherwise reconstruct surface text using multiword-token forms and spacing metadata, excluding empty nodes. Never use gold labels in fingerprint construction. Whitespace collapsing applies only to fingerprint copies and must not erase paragraph boundaries during segmentation. Case or punctuation revisions remain within the documented near-duplicate limitations.
* Run these checks automatically whenever corpus inputs change. Record input content checksums, normalization version, and exclusion counts automatically with the run; no manually maintained manifest is required. Freeze the resulting corpus identity for the experiment.
* Exact-match filtering does not establish freedom from near-duplicate passages or shared-source contamination. Known literature sources overlapping validation/test sources must be excluded in full, and remaining limitations must be reported.

### Apple Silicon Performance & Memory Guardrails

1. **Streaming & Memory Efficiency:**
* Read literature incrementally using bounded byte buffers, defaulting to 64 KiB (65,536 bytes). Segment and process complete sentences in literature inference batches bounded by token count, defaulting to 4,096 tokens per batch. File count is not a memory bound.
* Never accumulate an entire book, corpus, feature collection, or marginal-probability collection in memory during preprocessing or pseudo-label inference. Write accepted pseudo-labels incrementally under `data/interim/`. Use disk-backed duplicate indexes if their size would exceed the preprocessing budget.
* Apply the paragraph, sentence-character, sentence-token, and batch-token limits defined in the Raw-Text Segmentation Contract. Reject oversized literature candidates and record their counts; do not truncate sentences or join fragments. Gold sentences must remain intact. If gold data cannot be processed within the configured resource budget, fail explicitly.

2. **CRF Optimization:**
* Limit feature extraction window to $w \in [-2, +2]$ relative to the target token to maintain fast iteration speeds during confidence-score inference loops.

3. **Training & Pipeline Memory Budget:**
* CRFsuite training is an in-memory operation; generators do not make its training dataset memory-bounded. Require a recorded `max_pseudo_tokens` limit and deterministic, seeded selection of complete pseudo-labeled sentences before fitting. By default, `max_pseudo_tokens` equals the number of supervised tokens in `train_gold`, giving a maximum pseudo-to-gold token ratio of 1:1. The ratio is configurable and its effective token cap must be recorded. Do not split sentences or exceed the cap to fill the budget, and never silently reduce training data in response to memory pressure.
* Require a configurable pipeline memory budget, defaulting to 8 GiB (8,192 MiB). Monitor aggregate resident memory of the pipeline's main process and worker processes, and stop a run that exceeds the budget. Do not publish an incomplete model as successful. Monitoring is a best-effort safeguard, not a guaranteed operating-system memory limit.
* Record the effective read-buffer size, batch token limit, sentence limits, pseudo-label token cap, selection seed, memory budget, and observed peak resident memory with the run.

## Additional Implementation Constraints

### Technical Verification Invariants

* *Check:* Does feature extraction respect the configured ±2 token window and sentence boundaries? Right context is permitted for offline tagging. CRF decoding and marginals may still depend on the complete sentence.
* *Check:* Does the CoNLL-U loader satisfy the Gold CoNLL-U Input Contract, preserve original files and UPOS labels, and produce aligned target labels using the same validated mapping across all splits and both models?
* *Check:* Does confidence filtering accept complete sentences exactly when every Viterbi-predicted label's marginal is at least the selected threshold, using unrounded probabilities and preserving all tokens, labels, and their order?

## Resolved Specification Decisions

**Resolved — validation-based threshold selection:** `val_gold` may guide selection of CRF hyperparameters and pseudo-label confidence thresholds, but may not supply training targets or learned preprocessing statistics. This distinguishes model selection from fitting and preserves test isolation. Repeated selection can overfit the development split; the threshold search must therefore be predefined and recorded.

The repository layout above is the prescribed blueprint, not a claim about the current filesystem.
