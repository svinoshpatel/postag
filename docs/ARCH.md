# Architecture and Technical Specifications

This document defines the technical requirements, experiment design, data contracts, and guardrails for the Ukrainian semi-supervised CRF POS tagger. See [AGENTS.md](../AGENTS.md) for operating rules, [PLAN.md](../PLAN.md) for goals/status and [HISTORY.md](HISTORY.md) for implementation history and verification evidence. The prescribed architecture does not imply that its components have already been implemented or verified.

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

#### Gold Parser Interfaces

- `iter_gold_sentences()` in `crf.data.conllu` yields immutable `GoldSentence` parser outputs; `to_record()`/`iter_gold_records()` produce validated normalized `SentenceRecord` objects without publishing artifacts. Only dataset records expose feature forms. Test-label access requires `final_evaluation=True` before source access.
- Parser and audit require a configured `InputIdentity` and repository root, verify opaque bytes before parsing and recheck at EOF. Exhaust iterators before accepting a source: later errors invalidate earlier yields.
- `iter_gold_audit()` returns a separate label-free type using only IDs, FORM, selected surface/document comments and spacing metadata. The seven annotation columns are checked only for framing/nonemptiness; arbitrary original comments are retained only by labeled outputs.
- Strict framing precedes `conllu` ID/comment helpers. Ranges precede their words, cover at least two words, do not overlap and end within the sentence. Decimal IDs are unique and follow their observed anchor (or anchor zero). This is POS parsing, not dependency-tree validation.
- Accept LF/CRLF and extra blank lines; reject a file-start BOM, invalid UTF-8, comments-only sentences and missing final delimiters with path/physical-line diagnostics.
- Retain surface-row IDs/forms and `SpaceAfter`/`SpacesAfter`/`SpacesBefore`. Reconstruction emits each MWT surface once instead of its words, skips empty nodes, uses escaped spacing or `SpaceAfter=No`/default inter-token spaces, and omits default trailing space. Supplied `# text` takes precedence as original text; otherwise mark reconstructed. Document/paragraph markers persist; never infer groups from `sent_id`. Gold-word spans remain null; gold is not filtered by raw-text limits. Orchestration owns memory monitoring and artifact publication.

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

#### Target Label-Space Design

- `crf.data.labels.TagMapping` centralizes immutable canonical entries, source validation/mapping, target validation and sorted inventory. `PipelineConfig.label_space`/`target_inventory` expose it; configuration keeps tuple entries and its existing serialized object. `crf.utils.config.UPOS`/`SERVICE_TAGS` remain compatible re-exports.
- Reject duplicate/invalid direct tuple entries; custom targets preserve exact case/Unicode and require valid UTF-8 without whitespace/NUL or a UD-tag collision. Gold labels are validated before mapping; pseudo targets are checked directly, never remapped. Audits do not access label-space logic.
- Preserve original forms/labels/files. Configuration failures remain `ConfigError`, dataset failures `ValidationError`, and gold-row failures retain path/line diagnostics. Existing configuration hashes and dataset target arrays preserve mapping identity without schema changes; derive inventory from configuration. Model/metric/prediction metadata must persist/check that inventory. No post-hoc mapped predictions or confidence aggregation is provided.

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

#### Baseline Search Interface

`crf.models.search.search_baseline()` consumes validated supervised-training and validation records, rejects other roles before fitting, and freezes the configured Cartesian grid in ascending `(c1, c2)` order. The default grid contains nine fits. Reserve a fresh `reports/<run_id>/baseline/` directory; persist exact ordered gold training records at `data/processed/<run_id>/baseline_train.jsonl`. Gold records remain in memory for repeated native fits; retain only one fitted estimator at a time. Compute one train-only OOV vocabulary and reuse the same validation records and masks for every candidate. Use stable `baseline_000`-style trial/candidate IDs; save each fitted candidate and its validation report for later gold-only control/empty-augmentation reuse. Reuse consumes these original references/metrics rather than refitting or changing metadata. This interface does not resume a partial search.

Persist the baseline-only terminal ledger at `reports/<run_id>/baseline/trials.jsonl`; M7 owns the combined baseline/final `trials.jsonl` and frozen selection. On handled candidate failure, record that candidate as failed and every remaining configuration as not executed, publish the unsuccessful ledger where possible, and raise without returning a gate. Abrupt termination or ledger-write failure leaves an incomplete search. Select only from complete successful grid coverage, using descending unrounded validation accuracy, descending macro-F1, then ascending `(c1, c2)`. Return the exact selected persisted candidate reference plus all cached trial outcomes, never a newly refitted estimator. A baseline result alone does not authorize final evaluation or publish `selected.json`/`run.json`.

#### Single-Candidate Training Interface

`crf.models.training.fit_candidate()` accepts an ordered iterable of validated `SentenceRecord` objects, a configuration, and one `(c1, c2)` from its frozen grid. Gold-only is the default; augmentation requires explicit `allow_pseudo=True`. Reject held-out roles in either mode and require at least one supervised sentence. Features receive only normalized forms. Consume the entire iterable before fitting; hash its canonical `jsonl_bytes(record.to_dict())` bytes in supplied order. Later dataset publication must use those same bytes. Do not infer an original source length or reopen corpora here; source/content/span validation remains the record producer's responsibility.

Only one fit may be active per process, on the pipeline main thread. The wrapper creates a fresh estimator and monitors feature preparation, native data loading and optimizer messages. It owns an 8-GiB-by-default configured monitor unless an active caller-owned monitor is supplied. With a shared monitor the result remains provisional until successful enclosing monitor exit and `ensure_success()`; no model/run publication occurs here. Failed fits close the tagger and remove only their library-owned temporary model. Successful estimators retain their temporary native file until serialization or disposal; they contain no monitor or callback closure.

Read the pinned CRFsuite optimizer messages independently of its lossy parsed loss fields: require finite losses, consecutive completed iteration numbers and one recognized successful stop. Explicit maximum-iteration termination is `iteration_limit`; stopping-criteria/convergence messages are `converged`. Unknown/missing stops, optimizer errors (even if the library writes a model), or inconsistent iteration counts fail. Configured `max_iterations=0` retains CRFsuite's unlimited convention. Training records retain complete effective estimator options (including explicit line-search defaults), ordered dataset checksum/counts, configuration hash, seed, Python/Unicode and installed dependency versions. L-BFGS uses no random generator or `random_state`; seed is recorded without changing global random state. Sequential execution and seed recording do not guarantee bitwise results across platforms or native-library versions.

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

#### Raw reader and adapter boundaries

The reusable reader emits immutable, label-free raw sentences, not pseudo-labeled `SentenceRecord` objects; targets and confidence metadata are added only after inference. The same paragraph adapter is usable without literature discovery for standalone inference. Discovery validates gold paths/physical identities to reject aliases but never opens gold contents. Content-based gold-export detection belongs to the separate contamination audit.

Adapter version 1 uses the pinned non-legacy `uk_two` sentence rules before adapting word spans. Preserve all nonwhitespace source content, including formatting characters omitted by the upstream word API; recover forms from aligned source slices rather than upstream canonicalized strings. Split punctuation individually and compound hyphens (`-`, U+2010, U+2011); preserve apostrophes between letters (including preceding combining stress marks) and numeric internal decimal/date/time separators (`.`, `,`, `:` between digits). Group ASCII digits separated by ordinary spaces into thousands groups (one to three leading digits followed by groups of three, with an optional decimal fraction); normalization admits NBSP variants. These bounded rules are not a full UD tokenizer for URLs, unusual numeric notation or MWT expansions. No gold boundaries are changed.

Paragraphs comprise nonblank physical lines with their original intervening line endings; blank lines contain only Unicode whitespace (an optional file-start BOM is ignored for blankness). A paragraph excludes its final line terminator and delimiter lines. CRLF is one line terminator but two original code points. Limits count the retained original paragraph/sentence span before normalization; rejected paragraphs retain only counts while draining to their delimiter. Paragraph ordinals include oversized paragraphs; sentence ordinals count segmented nonempty candidates, including rejected sentences, across each file. An oversized paragraph has no inferred sentence count. Exclusions are yielded incrementally with reason, source offsets and text volume, so callers can aggregate diagnostics without retaining text.

Exhaust source iterators before accepting a run: checksums are verified before decoding, against consumed bytes at EOF and again against the current source. A late decoding/integrity failure invalidates earlier yields. Decoded offsets index the text produced by the configured strict codec (a BOM-consuming codec, if explicitly selected, defines that decoded coordinate system). The default UTF-8 codec retains the BOM for shared normalization to handle.

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
* **Sampling storage/API:** `crf.models.sampling` accepts validated gold records and contamination-approved confidence candidates, reserving a fresh `data/interim/<run_id>/sampling/` SQLite index. Interpret the configured ratio through its decimal string for cap arithmetic, avoiding binary-float boundary loss and multiplication overflow. Indexed `(priority, fingerprint)` ordering and file-backed SQLite temporary storage avoid whole-pool memory accumulation. Duplicate candidate fingerprints/identities fail rather than silently selecting a stream-dependent representative. Each threshold publishes one no-clobber `data/processed/<run_id>/extended_<index>.jsonl` dataset and returns an immutable path/checksum/count reference for reuse by every fit. Failed calls may leave partial intermediates/datasets but return no authorized selection; experiment completion remains orchestration-owned. Empty selections contain only gold records and must be treated as baseline reuse by orchestration.
* **Selection diagnostics/API:** The sampler writes `reports/<run_id>/pseudo_selection.json` with received/eligible/selected sentence and token counts, confidence/budget exclusions, cap utilization (null for zero cap), and source, sentence-length and predicted-POS distributions/rates. `iter_scored_candidates` retains scores for every contamination-eligible sentence for complete confidence denominators; the existing minimum-threshold pool API remains supported, with reports explicitly marking its limited denominator coverage. Optional completed contamination audit metadata is attached separately and validated against configuration and full-stream counts; absence is reported as unavailable, never zero exclusions. Summary group counts use disk-backed tables; full sentence bodies below the minimum threshold are not retained. No automatic threshold lowering or balancing occurs.
* **Empty reuse API:** `reuse_empty_selections` verifies complete baseline grid coverage, same-run/gate identity, frozen gold-only dataset bytes/counts, candidate metadata/model checksums and cached validation reports. It publishes `reports/<run_id>/empty_reuse.json`, containing explicit final `reused` trial records with original model/dataset references and unchanged cached scores. It never fits or re-evaluates. Nonempty thresholds remain for M7 fitting. When all thresholds are empty it records that self-training was not performed and retains the baseline gate, with no augmentation gain claim. The helper does not mark development complete or replace M7 combined trial accounting.
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

### Legacy Interface Compatibility

The auxiliary `scripts/fetch_dictionary.py` utility remains outside the CRF feature pipeline. Its default manifest is `data/dict_uk/source.json`; `--source` may select another direct-child manifest in that same directory. A missing manifest is an explicit error, not permission to infer a release. Reusable retrieval belongs in `crf.utils.dictionary`, with a thin argparse CLI. Require a single safe filename, pinned SHA-256 and credential-free HTTPS URL; reject insecure redirects and symlink/nonregular destinations. Anchor reads and publication to a directory descriptor without following symlinks. Preserve verified existing files and publish fresh downloads without replacement, cleaning temporary files after failure. Existing valid HTTPS manifests remain supported; unsafe paths/URLs are intentionally rejected. Downloads stay under `data/dict_uk/` and are Git-ignored; the default source manifest may be versioned. Dictionary downloads do not authorize writing into immutable `data/raw/` or consuming dictionary features in the model.

Use one `crf` production interface with dictionary features and bounded preprocessing. Legacy imports, keys, CLI switches and exploratory model artifacts are unsupported because corpus-frequency/±1 features, SVD/clustering and whole-file spaCy processing conflict with those contracts.

Keep the low-level tokenizer/preview separate from the production adapter. It accepts one supplied paragraph, uses non-legacy upstream spans and preserves original slices; the preview emits text-only JSON to stdout without training artifacts. The production adapter must integrate aligned normalization, UD boundaries, bounded file streaming, exclusions and provenance. Migration history is in [HISTORY.md](HISTORY.md#milestone-1).

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

### Run Lifecycle Design

- `allocate_run()` reserves UUID-based ASCII IDs across all artifact trees, retrying collisions without reuse. Development creates report/interim/processed/model directories; prediction creates reports only. Partial failures leave incomplete directories, with no automatic resume/rollback. Random run IDs do not affect seeded data/model ordering.
- Writers traverse directories without following symlinks, stage fresh temporary files, flush/fsync and validate them, then hard-link the final name without replacement. Concurrent writers have one winner; preserve existing files/directories/links and clean only the writer's temporary entry. Unsupported POSIX operations fail explicitly. This guards ordinary races/path redirection, not hostile repository renaming or guaranteed power-loss durability.
- `publish_completion()` requires stage-specific schema/integrity validation and a finished memory monitor; validate the encoded snapshot, run ID, status/error and observed peak. Success requires a successful monitor context, including no body exception. `load_completed()` applies the same validator and rejects missing/failed/invalid completion; directory existence is insufficient. Callers must verify all stage fields, checksums, candidate links, trial counts and metrics. Failure descriptors may report latched resource failures but cannot authorize success.
- `configure_logging()` replaces one stderr handler on the nonpropagating `crf` hierarchy, preserving unrelated loggers; CLI results use stdout.
- `MemoryMonitor` is mutable because samples, peak/thread state and the first failure evolve. Sample aggregate root/recursive-worker RSS once per process at entry, background intervals (default 0.1 s), checkpoints and exit. Tolerate exited children; root/access/monitoring errors fail closed. Budget equality is allowed; exceedance/failure is permanently latched. Background failure interrupts the main Python thread; checkpoints/exit raise it.
- Pipeline owners must run monitoring on the main thread, clean owned workers in `finally` and guard completion. The foundation neither kills unrelated processes nor publishes failures automatically. Native code can delay interruption, spikes can escape sampling and shared pages can be double-counted: monitoring is best effort, not an OS memory limit.

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

#### Dataset and Identity Interfaces

- Immutable `SentenceRecord`, `PseudoMetadata` and `InputIdentity` use validated factories. Dataset validation requires configuration/input identity and original decoded length for any non-null span; never infer offsets from sentence text/annotation rows. IDs are `1..N`, spans remain ordered/disjoint across nulls, features receive only `feature_forms`, and test records require final-evaluation opt-in.
- Validate schema, mapping/inventory, provenance IDs, input bindings, seed-derived priorities and exact shared normalization of original forms. Fingerprint/gate hashes are syntax-checked here; content/model verification belongs to contamination/model stages. Loader comments/reconstruction metadata stay separate.
- Strict JSON rejects nested duplicate keys, nonfinite/overflowing numbers, BOMs and invalid Unicode. JSONL streams objects with path/line diagnostics, rejecting blank lines/CRLF; a final object may omit LF on input, but encoders always append it. Encoders return bytes without publishing files; lifecycle helpers own publication, and later stages own their specific schemas.
- Discover identities in gold-role order, then flat filename-sorted eligible literature. Validate paths/symlinks, ignore directories/non-`.txt` entries, reject physical gold aliases and empty literature. Hash bounded original-byte chunks, treating test content as opaque; check metadata before/after and reject observed changes. This is best-effort consistency, not a snapshot. `verify_input_identity()` rechecks disk bytes; content overlap/known-source audits are separate preprocessing responsibilities.

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

#### Bundle Persistence Interface

`crf.models.bundles.save_bundle()` reserves a fresh candidate directory at the prescribed model location. It requires the successful finished training monitor and uses a separate configured monitor for serialization and round-trip checks. This permits sequential candidate publication without retaining every estimator until development finishes. Later run completion still requires its own successful monitoring and all trial links. Publish `model.joblib` without replacement only after serialization checks and successful monitor exit; publish `metadata.json` last. Failure may leave an incomplete directory or model file, never a loadable pair. Do not resume or overwrite such directories.

Sidecars contain the complete canonical effective configuration, not a partial settings document. `parse_config(resolve_paths=False)` checks recorded paths lexically, without resolving or inspecting the original corpus filesystem; ordinary configuration loading retains its existing resolved-path checks. Require exact installed Python, Unicode and runtime-dependency versions, full estimator options consistent with configuration/grid, inventory, seed, counts and termination evidence. A model may have learned only a subset of the fixed target inventory. Persist the immutable training record inside the estimator as well as the sidecar and require agreement, binding counts/dataset identity/configuration to the serialized estimator.

`load_bundle()` accepts a standalone copied directory without the original repository or datasets. Optional expected model/metadata checksums and run/candidate IDs bind frozen references; when a repository root is supplied, require the prescribed candidate location as well. Open regular files through directory-anchored no-follow descriptors; reject symlinks and special files. Stream the native artifact into an anonymous temporary snapshot, verify its checksum before deserialization, and load those verified bytes. Validate estimator type/options, embedded training evidence, learned labels and raw optimizer termination before returning it. Loading creates anonymous snapshots and dependency-owned native temporary files, never repository artifacts. Close/dispose rejected estimators.

Save/load verification uses fixed synthetic surface-form probes through the recorded feature/normalization versions; require identical predictions and all unrounded token marginals, including each Viterbi label's confidence. Synthetic probes do not access gold/literature or constitute model evaluation. Joblib uses pickle: these are trusted local artifacts; checksum/schema validation detects mismatches and corruption, not malicious code or authenticity. Native persistence and RSS monitoring remain best effort and exact environment matching is intentionally conservative.

### Development Search Interface and Staged CLI Migration

`crf.pipeline.develop` allocates a fresh run, records canonical configuration and opaque input identities, runs baseline selection, contamination-approved full-confidence inference, threshold sampling and verified empty reuse, then fits each nonempty final candidate from scratch. Frozen threshold datasets are streamed/validated on every fit, with exact byte identity checked before native optimization. Gold order and source snapshots remain fixed; only train/validation labels enter model APIs. Gate and final estimators are disposed between stages/candidates. Baseline search can expose its completed outcomes through an optional callback before ledger publication so an enclosing run retains actual fit outcomes even if that publication fails.

The combined `reports/<run_id>/trials.jsonl` accounts for every planned baseline/final configuration, including failed and unexecuted outcomes on handled aborts. No automatic partial-run resume is provided. `search_development` is the search-only library primitive and cannot authorize evaluation. `develop` captures start-time source/environment provenance, monitors the entire search/freeze stage, validates persisted evidence, exits monitoring successfully, then publishes selection and `run.json`. The CLI reports `complete` only for that authorized result; handled failures publish a failed descriptor where possible, otherwise leave an incomplete run. The earlier task-1 pending-freeze CLI status is superseded.

`crf.experiments` strictly parses trial, selection and completion schemas; checks full frozen-grid coverage; validates canonical configuration, source snapshot ordering/current opaque identities, exact dataset bytes/order/counts/caps, pseudo fingerprints/confidences/gate provenance, every model/sidecar checksum/training history and cached validation report; and recomputes gate/final/control ranking. Reused trials retain their corresponding baseline history and scores. Frozen loading repeats these checks through `load_development`, without deserializing estimators or accessing test labels, and rejects failed/missing/changed artifacts. Auxiliary sampling/audit diagnostics remain reports, not alternative completion authorities. Source Git revision/dirty state is observational provenance, not a requirement to have a clean checkout or the same HEAD when reading a frozen run. Read-only loading does not write a new manifest or mutate history.

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

`crf.evaluation` provides validation-only scoring of validated gold records and immutable sentence predictions carrying sentence IDs, token IDs and normalized forms. Build one immutable lexical vocabulary from `train_gold` and reuse it across candidates; bind it to the effective configuration. Reject held-out test roles, duplicate/missing/extra/reordered predictions, misaligned forms/IDs and incompatible labels. This layer performs no inference and opens no corpora. Final test orchestration remains a later interface. Report validation recomputes primary metrics from confusion counts with floating-point tolerance; diagnostic membership is established during aligned scoring, not recoverable from aggregate report counts. Publication requires successful finished monitoring and uses the existing no-clobber artifact writer. Frozen-reference checks against actual bundles and trial-history links belong to orchestration.

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

## 4. Environment & Packaging Contract

Declare direct runtime dependencies and development tools in `pyproject.toml`; reproduce the environment with `uv.lock` in the project-local environment. Include Ruff, mypy, and pytest as project development tools. Keep Ruff configuration in `pyproject.toml`, target Python 3.14, and document material lint-rule choices. See the [uv project structure reference](https://docs.astral.sh/uv/concepts/projects/layout/).

Select compatible dependency versions through `pyproject.toml` and `uv.lock`; reproduce the environment with `uv sync --locked`. Environment acceptance requires imports, a tiny CRF fit/marginal/persistence round trip and an isolated installed-package import. These checks do not establish real-corpus or cross-platform reproducibility; execution evidence belongs in [HISTORY.md](HISTORY.md#verification-record).

Ruff enables essential syntax/error checks (`E4`, `E7`, `E9`, `F`), import ordering
(`I`), Python modernization (`UP`), and common correctness pitfalls (`B`).
Formatting remains a separate check. Mypy uses strict mode for `src/crf/`.
Pytest uses importlib mode and the installed package, with no `sys.path` injection;
integration tests carry an explicit marker and document their dependencies.
`src/crf/py.typed` marks the installed package as typed. The Hatch build backend
packages only `src/crf/`.

Seeds control applicable random operations but cannot guarantee identical floating-point results across platforms/dependency versions. Record the effective environment and observed nondeterminism with run artifacts.

Before producing artifacts, configure `.gitignore` to exclude generated datasets, model bundles, and run outputs. Original datasets remain immutable and generated artifacts must not be staged or committed, except for intentional small test fixtures permitted by the repository operating rules.

The required verification workflow and concrete commands are defined in [AGENTS.md](../AGENTS.md#4-verification-and-test-generation); dated results belong in [HISTORY.md](HISTORY.md#verification-record).

### IDE Environment Design

Use the project-local native uv SDK and inherited project interpreter; packaging/runtime authority remains `pyproject.toml`, `.python-version` and `uv.lock`. Disable the IDE's `usePyprojectToml` module model and maintain explicit source/test roots and exclusions to keep IDE layout aligned with the repository. See [PyCharm setup](PYCHARM.md) and [setup history](HISTORY.md#pycharm-environment).

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

#### Configuration Validation Design

- `load_config(path, repository_root=..., seed=None)` resolves against the explicit repository root. Immutable corpus paths are repository-relative `Path` values; `to_dict()` emits complete settings with POSIX paths and array grids. Numeric settings accept integers/floats and normalize to floats; integer fields require integers. Neither accepts booleans. Reject merge keys, recursive aliases, non-string keys and non-plain values; allow harmless nonrecursive aliases. Only version 1 schemas/preprocessing/features are supported.
- CRF domains: `max_iterations >= 0` (zero is unlimited), positive `num_memories`/`period`, finite nonnegative `epsilon`/`delta`/`min_freq`, boolean expansion switches. Overrides define separate experiments; defaults retain 36 configurations. Source codecs must support incremental text decoding.
- Loading checks existing `.conllu` gold files under `data/raw/gold/` with distinct device/inode identities (including hard links), and existing nonsymlinked `data/raw/literature/`. Roots remain inside the repository and disjoint; no labels are read. `parse_config()` checks settings/path syntax without requiring corpora, supporting standalone metadata. Development must call `load_config()` or `validate_input_paths()`; content eligibility/auditing is a separate preprocessing responsibility.
- `resolve_cli_path()` permits external standalone inputs. `validate_output_path()` checks fresh prescribed artifact paths and rejects existing/dangling entries, symlinked ancestors, escapes and raw destinations; it allocates nothing. Lifecycle helpers supply race-safe publication.

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

#### Normalization and Source Alignment Design

- `crf.data.normalization` version 1 records the running Unicode database version. `normalize_text()` takes a complete decoded paragraph/bounded unit and nonnegative absolute `source_offset`; it removes one BOM only at file offset zero and applies the ordered transformations above without lowercasing/collapsing whitespace. Raw-text streaming must assemble paragraphs without splitting CRLF or normalization units. The low-level tokenizer/preview is not that production adapter.
- Immutable results retain original/normalized text, per-character absolute source intervals, formatting observations and versions. Each output character maps conservatively to its entire NFC unit (starter/combining marks and interacting zero-class compositions, including Hangul), covering composition, expansion and reordering. `source_span(start, end)` projects nonempty in-bounds normalized intervals to original end-exclusive code-point spans, rejecting cuts inside units. CRLF maps to both characters; removed BOMs lie outside projected text. Adjacent projected tokens cannot overlap or lose combining marks.
- `normalize_form()` preserves a form's leading BOM and token boundary; `normalize_forms()` preserves order/length. The old surface-module import remains a re-export. Casing uses normalized forms; `lexical_form()` lowercases and reapplies NFC without case folding.
- Preserve `Cf` characters except the permitted file-start BOM. Results record every original offset/removal flag; raw/form normalization logs a summary. Stress marks remain under NFC. Empty raw text is valid; empty forms, invalid scalars, boolean/noninteger/negative offsets and unsupported versions fail. No filtering, truncation, filesystem access or corpus statistics occurs here.
- Gold conversion revalidates source/target/configuration and explicit test authorization through `parse_sentence_record()`, keeps null word spans and separate parser metadata. Gold/pseudo normalized forms must exactly match shared normalization. Schema stays version 1; incorrect older normalization is rejected. Later contamination/literature/inference must use this API and its offsets; tokenizer compatibility/corpus readiness are separate gates.

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

#### Feature Validation Design

- `crf.features.schema` owns version 1, aliases, affix/context constants and validation; configuration shares its version check. `sentence_features(forms, schema_version=1, normalizer_version=1)` preserves default calls. Versions are keyword-only and training/inference callers must pass recorded settings; they cannot change the window/features. Existing surface-module alias/normalizer imports remain compatible; no dependency or wire-format change is required.
- Accept only nonempty ordered surface-string sequences; reject sentence strings, mappings, generators, token-label pairs and annotation-bearing records. Shared normalization snapshots forms before shape computation. Each token/call gets an independent mutable dictionary because CRFsuite needs dynamic affix/boundary mappings; there is no vocabulary fitting, shared cache or corpus state. Consumers pass only `SentenceRecord.feature_forms`.
- `validate_sentence_features()` returns copies of a nonempty mapping sequence without coercion/drop. Enforce exact position/length-dependent keys, strict string/boolean/finite-float values, `bias: 1.0`, true boundary flags, normalized lexical forms, affix contents and neighbor agreement. Context has lexical/six-flag data or one boundary flag, never both or neighbor affixes/bias. Reject unknown/missing keys, invalid Unicode, nonfinite floats and numeric booleans with token/key diagnostics. Internal consistency cannot recover original casing; the extractor computes shape semantics from normalized surface forms.
- Extraction/validation raise `FeatureSchemaError` (`ValueError`); configuration raises `ConfigError`. Tokenizer/corpus compatibility is a separate acceptance gate; execution evidence belongs in [HISTORY.md](HISTORY.md#verification-record).

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

#### Contamination Audit Interface and Policy

The pseudo-label preprocessing filter owns a fresh SQLite index under `data/interim/<run_id>/`, reserved without replacement using the artifact lifecycle helper. It is an incomplete intermediate, never proof of a completed run. It is rebuilt from every configured input snapshot; it is not reused after an input change. SQLite indexes store gold fingerprints, train-only MWT boundary patterns and first eligible literature occurrences on disk, with a bounded page cache and file-backed temporary storage. Gold is streamed through `iter_gold_audit()` only, without labeled records or annotation interpretation. Gold sentence text takes precedence over reconstruction as prescribed above; all gold sentences are audited intact regardless of raw-text size limits.

Fingerprint schema version 1 applies the configured shared normalizer to sentence text without removing an embedded leading BOM, then collapses Unicode whitespace and trims before SHA-256. Case, punctuation and formatting characters remain significant. Raw-file BOM handling has already occurred in the reader. Empty/whitespace-only fingerprint inputs fail. Never construct sentence text by joining syntactic words: that would lose MWT surfaces and spacing metadata.

Known-source evidence cannot be inferred from filenames or sentence IDs. The filter accepts explicit immutable exclusion rules binding an eligible literature path to `val_gold` or `test_gold` with a nonempty provenance explanation. Validate rules against discovered paths, reject duplicates/unknown paths, and include the canonical rules in automatic audit metadata and corpus identity. This is an exclusion policy, not an input-discovery manifest; adding literature still requires no manual source listing. In the absence of established provenance, an empty rule set is allowed for fixture-based work but does not establish real-corpus source separation.

MWT exclusion uses only multiword range-row surface forms observed in `train_gold`. Normalize and tokenize each surface with the same versioned adapter and store its complete normalized token sequence. Reject a literature sentence containing that contiguous token sequence, even if spaces around punctuation differ. This conservative boundary-aware match handles adapter-split hyphens and avoids substring matches inside larger words; it preserves case and performs no expansion or label interpretation. Validation/test MWT inventories must not affect this rule.

Precedence is full-source exclusion before decoding, then raw resource exclusions, gold exact overlap, known train MWT, then literature duplication. Keep only the first otherwise eligible sentence for each fingerprint in filename/sentence discovery order. Excluded sources/sentences never reserve a duplicate slot. Diagnostics retain the reason and applicable split/MWT/first-occurrence/evidence information; source exclusions report byte volume without decoding an entire excluded file. Resource diagnostics retain their original character/token counts. The filter is pseudo-label-specific; standalone prediction continues to use the unfiltered raw reader.

Audit iterators are single-use. No report or frozen corpus identity is available until every input has been consumed or excluded, the current discovered identities match the initial snapshot, and all source checks have succeeded. Reports expose versions, effective configuration hash, input checksums, exclusion rules/counts, accepted counts and deterministic accepted-corpus identity for later run publication. Early termination, decoding, parsing, identity or SQLite failure invalidates the audit; original datasets are never modified. Near-duplicate and unresolved source-provenance limitations remain explicit in the report.

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

#### Preprocessing Monitor Ownership

Source hashing, gold/raw readers, sentence batching and contamination auditing accept an optional caller-owned `MemoryMonitor`. Existing unmonitored library calls remain compatible for bounded exploration; a pipeline owner must enter a configured monitor around production preprocessing, consume iterators fully and call `ensure_success()` after monitor exit before publishing successful outputs. Readers neither start/stop the shared monitor nor silently reduce data. Configuration-aware calls reject a monitor whose budget differs from the effective configuration, and reading with an inactive/finished monitor fails.

Checkpoints occur on bounded source-byte reads (including opaque identity hashing and draining oversized literature), gold physical lines and complete-sentence boundaries, adapted raw candidates, batch boundaries and audit construction/selection/final identity checks. A memory or sampling failure propagates immediately as a resource failure; it is never counted as a resource-size exclusion or a reason to skip/truncate gold. Reports obtained while the enclosing monitor is active remain provisional until that monitor completes successfully. An audit report checks a still-active monitor, or requires success from an already-finished monitor. Database/intermediate presence alone never authorizes pipeline completion. Background sampling remains the best-effort guard during indivisible normalization/tokenizer/native operations; checkpoints do not establish a hard memory limit.

## Additional Implementation Constraints

### Technical Verification Invariants

* *Check:* Does feature extraction respect the configured ±2 token window and sentence boundaries? Right context is permitted for offline tagging. CRF decoding and marginals may still depend on the complete sentence.
* *Check:* Does the CoNLL-U loader satisfy the Gold CoNLL-U Input Contract, preserve original files and UPOS labels, and produce aligned target labels using the same validated mapping across all splits and both models?
* *Check:* Does confidence filtering accept complete sentences exactly when every Viterbi-predicted label's marginal is at least the selected threshold, using unrounded probabilities and preserving all tokens, labels, and their order?

## Resolved Specification Decisions

**Resolved — validation-based threshold selection:** `val_gold` may guide selection of CRF hyperparameters and pseudo-label confidence thresholds, but may not supply training targets or learned preprocessing statistics. This distinguishes model selection from fitting and preserves test isolation. Repeated selection can overfit the development split; the threshold search must therefore be predefined and recorded.

The repository layout above is the prescribed blueprint, not a claim about the current filesystem.
