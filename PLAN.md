# Sprint Plan

Goals, prerequisites, acceptance criteria and task status. See [AGENTS.md](AGENTS.md) for operating procedures, [docs/ARCH.md](docs/ARCH.md) for design/contracts and [docs/HISTORY.md](docs/HISTORY.md) for build history and verification evidence. Mark goals complete only after acceptance and applicable verification.

## Sprint Goal and Progress

Implement a reproducible Ukrainian CRF self-training pipeline with isolated development and final evaluation, usable raw-text inference, and validated run artifacts. Preserve the approved experiment design and original datasets.

Completed goals: M1–M7 (fixture-based acceptance). Next: M8 frozen evaluation and standalone prediction. Production-corpus readiness and end-to-end pipeline completion remain separate goals; evidence is in [HISTORY.md](docs/HISTORY.md).

Gold-corpus availability, release identity, and compatibility checks are pending. Small deterministic fixtures support implementation and offline verification without production data. Full corpus runs have a separate readiness gate below; missing inputs must not be replaced with another split or fabricated data. No sprint duration or delivery estimate is assumed.

## Ordered Implementation Milestones

### Refactoring and Refinement — Existing M1–M4 Implementation

Scope: preserve public preprocessing/configuration contracts and existing user changes; improve demonstrated safety, bounded work and readability before M5. Evidence: [refactoring review](docs/HISTORY.md#refactoring-and-refinement).

- [x] Make memory-monitor scheduling fail closed and split configuration section parsing into focused typed helpers. Acceptance: invalid intervals and scheduling failures cannot authorize success; effective configuration values and hashes stay compatible.
- [x] Replace repeated MWT candidate scans with indexed exact probes and clarify contamination decisions. Acceptance: exclusion priority, matched surfaces, counts and corpus identities remain stable; matching remains interruptible with bounded Python encoding/probe work.
- [x] Harden the legacy dictionary downloader and reduce its CLI to reusable typed logic. Acceptance: metadata cannot redirect writes, existing files are never replaced, failed downloads clean temporary files, and offline tests exercise failures without network access.
- [x] Review each implementation against the specification before new tests, then run focused regressions and all four required checks. Acceptance: actual results and compatibility limits are recorded in history.

Milestones follow the prerequisites below. Execution procedures are in AGENTS.md; requirement reviews and verification records belong in HISTORY.md.

### 1. Establish the Environment and Core Package

Architecture: [Core Package and Migration Boundary](docs/ARCH.md#core-package-and-migration-boundary) and [Environment & Packaging Contract](docs/ARCH.md#4-environment--packaging-contract).

- [x] Establish the prescribed CPython/`uv` environment, dependency declarations, lockfile, interpreter selection, and Ruff/mypy/pytest configuration. Acceptance: dependency compatibility is demonstrated and all four verification tools are runnable from the repository root.
- [x] Create installable `src/crf/` packaging and migrate or replace relevant `postag/` logic and affected tests/callers. Acceptance: pipeline imports use `crf`; reusable logic resides in the core, with no dependency on importing `src` as a package or preserving incompatible exploratory behavior.
- [x] Add required generated-artifact exclusions before producing datasets, models, or reports. Acceptance: immutable inputs and existing artifacts are protected; intentional test fixtures remain identifiable.

Evidence: [M1 history](docs/HISTORY.md#milestone-1), including the permitted [lint/format exception](docs/HISTORY.md#recorded-limitations).

#### PyCharm Environment Goals

- [x] Align project/module roots and inherited interpreter with the uv-managed Python 3.14 environment; preserve existing project-model changes.
- [x] Register the existing environment as a native uv SDK, add current preview/test run configurations, and document reload/setup behavior.
- [x] Validate XML/path references, interpreter/import compatibility, and all four required checks; record IDE validation limits and pre-existing failures.

Evidence and remaining UI-validation limit: [PyCharm history](docs/HISTORY.md#pycharm-environment).

### 2. Implement Shared Configuration, Artifact, and Resource Foundations

Prerequisite: milestone 1. Architecture: [Artifact & Run Contracts](docs/ARCH.md) and [CLI & Configuration Contract](docs/ARCH.md).

- [x] Implement strict YAML configuration validation, documented defaults, seed precedence, version checks, and repository-relative path resolution. Acceptance: invalid/duplicate/unknown fields and incompatible settings fail explicitly; gold split aliases and source/output path violations are rejected.
- [x] Implement typed sentence records, JSON/JSONL validators, canonical configuration hashing, deterministic sentence identities, and automatic input identities. Acceptance: exact schema fields, types, checksums, alignment, and nullability are enforced without exposing labels to feature code.
- [x] Implement unique output allocation, publish-once completion handling, logging, and aggregate RSS monitoring. Acceptance: existing artifacts and raw inputs cannot be overwritten, diagnostics remain on stderr, budget violations fail, and incomplete runs cannot be treated as successful.

Evidence: [M2 history](docs/HISTORY.md#milestone-2), including the permitted [lint/format exception](docs/HISTORY.md#recorded-limitations).

### 3. Implement Gold Parsing, Target Mapping, and Features

Prerequisites: milestones 1–2. Architecture: [Gold CoNLL-U Input Contract](docs/ARCH.md#gold-conll-u-input-contract), [Optional Target Tag Mapping](docs/ARCH.md#optional-target-tag-mapping), and [Ukrainian Language & NLP Feature Constraints](docs/ARCH.md).

- [x] Implement the streaming CoNLL-U loader and separate label-free audit interface. Acceptance: supplied sentence order and integer-word boundaries are preserved; multiword/empty-node rows, malformed records, final delimiters, provenance, and diagnostics follow the parser contract.
- [x] Implement optional service-tag mapping. Acceptance: original files and labels remain intact; source labels are validated first; all stages use the same configured target inventory; invalid mappings fail.
- [x] Implement the shared versioned normalizer with source alignment. Acceptance: apostrophes, Unicode normalization, casing, whitespace, and Cyrillic preservation match the architecture, with original forms retained.
- [x] Implement the feature dictionary schema and sentence-bounded context extraction. Acceptance: feature names/types, affix lengths, casing/shape flags, boundary flags, unseen forms, and annotation exclusion match the specification.

Evidence: [M3 history](docs/HISTORY.md#milestone-3), including the permitted [lint/format exception](docs/HISTORY.md#recorded-limitations).

### 4. Implement Literature Streaming and Contamination Controls

Prerequisites: milestones 2–3. Architecture: [Raw-Text Segmentation Contract](docs/ARCH.md#raw-text-segmentation-contract) and [Strict Test Data Isolation & Leakage Prevention](docs/ARCH.md).

- [x] Implement flat literature discovery, source encoding, and the versioned tokenizer/UD adapter. Acceptance: discovery never recurses or consumes gold; paragraphs and sentences survive buffer boundaries; original offsets and configured size limits are respected.
  - [x] Share guarded flat discovery with input identities without reading gold contents.
  - [x] Add bounded strict decoding/paragraph streaming and aligned versioned UD segmentation, with incremental exclusions and token-bounded batches.
  - [x] Review requirements, cover offline boundaries/failures, and run focused and all required checks.

  Evidence: [M4 task 1 history](docs/HISTORY.md#m4-task-1--literature-reader-and-ud-adapter). Types/tests and changed-code lint/format passed; repository lint/format retain only the unchanged `scripts/fetch_dictionary.py` failures. Full tests passed outside the sandbox after its process-enumeration restriction blocked the existing RSS smoke test. Official-corpus tokenizer compatibility remains pending.

- [x] Implement text-only gold overlap exclusion, deterministic literature deduplication, and known-source/MWT exclusions. Acceptance: automatic fingerprints and exclusion diagnostics follow the contract; held-out labels remain uninterpreted; original gold is unchanged.
  - [x] Add versioned surface fingerprints and a fresh disk-backed audit of every gold split using only the label-free interface.
  - [x] Filter literature in discovery order with full-source evidence rules, train-only MWT rules and first-eligible-occurrence deduplication.
  - [x] Record provenance/counts/corpus identity, review boundary/failure cases, and verify focused plus all required checks.

  Evidence: [M4 task 2 history](docs/HISTORY.md#m4-task-2--contamination-controls). Types/tests and changed-code lint/format passed; repository lint/format retain only the unchanged `scripts/fetch_dictionary.py` failures. Repaired the task-blocking committed tokenizer import typo. Real-corpus source provenance and near-duplicate limitations remain explicit.

- [x] Verify streaming and resource-failure behavior with bounded fixtures and controlled monitor inputs. Acceptance: oversized candidates are rejected intact, gold sentences are not filtered or truncated, intermediates are incremental, and inference-only filtering is distinct from pseudo-label selection.
  - [x] Thread an optional caller-owned memory monitor through source hashing, gold/raw streaming and contamination checkpoints without changing unmonitored calls.
  - [x] Verify whole-candidate limits, incremental indexes/output, bounded batches and inference/pseudo-label separation with offline integration fixtures.
  - [x] Verify controlled budget/monitor failures prevent success and preserve raw inputs; run all required checks and record limitations.

  Evidence: [M4 task 3 history](docs/HISTORY.md#m4-task-3--streaming-and-resource-failure-verification). Types/tests and changed-code lint/format passed; repository lint/format retain only the unchanged `scripts/fetch_dictionary.py` failures. Production-corpus compatibility and later pipeline/model orchestration remain separate acceptance gates.

### 5. Implement CRF Training, Model Bundles, and Validation Metrics

Prerequisites: milestones 2–3. Architecture: [Training & Model-Selection Contract](docs/ARCH.md), [Model Bundles](docs/ARCH.md#model-bundles), and [Metric Records](docs/ARCH.md#metric-records).

- [x] Implement the CRF training wrapper with the prescribed optimizer settings, sequential fitting, termination records, and memory failures. Acceptance: every candidate uses only its declared training records; iteration-limited models are not called converged.
  - [x] Add a single-candidate API with explicit gold-only/augmented input policy, fresh fits, ordered dataset identity and training/environment records.
  - [x] Validate optimizer termination and resource failures with offline fixtures, then run the four required checks.
  Evidence: [M5 task 1 history](docs/HISTORY.md#m5-task-1--single-candidate-crf-training). Fixture-based acceptance; production training remains pending.
- [x] Implement model serialization, metadata/checksum validation, and save/load verification. Acceptance: round-tripped predictions and confidences agree, mismatched bundles fail, and standalone loading does not require original training files.
  - [x] Add strict complete sidecars and portable loading with environment, parameter, inventory and checksum checks.
  - [x] Publish fresh candidate bundles after monitored round-trip verification; cover malformed/incomplete bundles and no-clobber failures, then run required checks.
  Evidence: [M5 task 2 history](docs/HISTORY.md#m5-task-2--model-bundle-persistence). Fixture-based persistence acceptance; real-corpus model production remains pending.
- [x] Implement validation metrics and reports. Acceptance: token accuracy, fixed-inventory macro/per-tag F1, confusion counts, shared gold-only OOV vocabulary, and punctuation diagnostics agree with deterministic fixtures, including empty diagnostic subsets.
  - [x] Implement aligned validation scoring, gold-only vocabulary and strict metric-report validation/publication; verify deterministic boundaries and all required checks.
  Evidence: [M5 task 3 history](docs/HISTORY.md#m5-task-3--validation-metrics). Offline acceptance; actual validation inference/search remains pending.
- [x] Implement the gold-only candidate search and gate selection. Acceptance: the nine baseline configurations, validation-only selection, tie breakers, cached candidates, and trial records match the contract; test predictions and metrics are unavailable during development.
  - [x] Add sequential baseline execution, persisted datasets/bundles/metrics and a complete terminal trial ledger, with fail-closed gate selection.
  - [x] Verify the default grid, tie breakers, isolation, retained candidates and failure accounting using offline fixtures and all required checks.
  Evidence: [M5 task 4 history](docs/HISTORY.md#m5-task-4--baseline-search-and-gate-selection). Default nine-fit search verified on tiny offline records; real-corpus experiments remain pending.

### 6. Implement Pseudo-Label Confidence and Sampling

Task 3 scope: record selection diagnostics with explicit denominator coverage and verify/cache baseline reuse for empty thresholds.

Prerequisites: milestones 4–5. Architecture: [Confidence & Sequence Acceptance](docs/ARCH.md) and [Pseudo-Label Sampling Contract](docs/ARCH.md#pseudo-label-sampling-contract).

- [x] Implement Viterbi-label marginals and complete-sentence acceptance. Acceptance: no token deletion or maximum-marginal substitution occurs; unrounded confidences and the minimum-grid-threshold candidate pool support every configured threshold.
  Evidence: [M6 task 1 history](docs/HISTORY.md#m6-task-1--viterbi-confidence-and-sentence-acceptance). Offline fixture acceptance; production inference remains pending.
- [x] Implement deterministic disk-backed priority ordering and whole-sentence sampling. Acceptance: the approved token cap is respected, oversized remaining-budget candidates are skipped without splitting, underfilled selections are valid, and each threshold's frozen dataset is reused across its fits.
  Evidence: [M6 task 2 history](docs/HISTORY.md#m6-task-2--deterministic-whole-sentence-sampling). Frozen dataset references are ready for reuse; fit orchestration remains M7.
- [x] Record acceptance, exclusion, source/POS distribution, and cap-utilization diagnostics. Acceptance: empty selections reuse corresponding baseline candidates and cached validation results without retraining or claiming augmentation gains.
  Evidence: [M6 task 3 history](docs/HISTORY.md#m6-task-3--selection-diagnostics-and-empty-baseline-reuse). Verified cached reuse and diagnostics offline; combined development orchestration remains M7.

### 7. Orchestrate Development and Freeze Selection

Task 2 scope: validate saved search/dataset/model/report identities, select final/control using frozen tie breakers, publish immutable selection/completion, and expose guarded completed-run loading.

Prerequisites: milestones 2–6. Architecture: [Frozen Model Selection](docs/ARCH.md#frozen-model-selection), [Automatic Run and Prediction Records](docs/ARCH.md#automatic-run-and-prediction-records), and [CLI & Configuration Contract](docs/ARCH.md).

- [x] Implement the `develop` CLI and final-model search. Acceptance: the default experiment accounts for nine baseline plus 27 final configurations; each final fit starts fresh unless baseline reuse applies; failed/unexecuted trials remain visible rather than shrinking the search.
  Evidence: [M7 task 1 history](docs/HISTORY.md#m7-task-1--develop-cli-and-final-model-search). Search primitive and CLI entry-point evidence; complete frozen authorization is covered by task 2 below.
- [x] Implement final selection, matched gold-only control, immutable run history, and frozen model references. Acceptance: selection uses the prescribed validation tie breakers, all linked identities/checksums agree, and only a complete successful development run authorizes final evaluation.
  Evidence: [M7 task 2 history](docs/HISTORY.md#m7-task-2--frozen-selection-and-development-authorization). Offline default/mixed/empty workflows and strict completed-run loading verified; production experiments remain separate.

### 8. Implement Frozen Evaluation and Standalone Prediction

Prerequisites: milestones 5–7 for final evaluation; milestones 2–5 for standalone prediction. Architecture: [Evaluation Contract](docs/ARCH.md#evaluation-contract), [Final Evaluation Reports](docs/ARCH.md#final-evaluation-reports), and [Standalone Prediction Reports](docs/ARCH.md#standalone-prediction-reports).

- [ ] Implement the `evaluate` CLI, three-model alignment, comparisons, and test reports. Acceptance: no refitting or overrides occur; changed/incomplete experiments fail; valid completed results are returned from cache; development history remains unchanged.
- [ ] Implement paired bootstrap and persisted resampling units. Acceptance: grouping/fallback, random generator, repeated-unit weighting, pooled accuracy, percentile interpolation, unavailable intervals, and integrity links follow the contract.
- [ ] Implement the `predict` CLI and fresh standalone reports. Acceptance: recorded model preprocessing is used, low-confidence and repeated sentences remain eligible, resource exclusions are reported, and custom mapped labels are not presented as standard UD UPOS.

### 9. Verify Integration and Complete Implementation

Prerequisites: milestones 1–8. Architecture: [Technical Verification Invariants](docs/ARCH.md#technical-verification-invariants); operating procedure: [Verification and Test Generation](AGENTS.md#4-verification-and-test-generation).

- [ ] Add and run deterministic integration coverage using small, intentional fixtures. Acceptance: the full develop/freeze/evaluate/predict flow, default search accounting, empty augmentation, isolation boundaries, schema/checksum rejection, and incomplete-output behavior are verified without production data or network access. Integration dependencies are documented.
- [ ] Run all four required checks and record actual results in [HISTORY.md](docs/HISTORY.md#verification-record). Acceptance: changed behavior is verified; any permitted unrelated pre-existing failures have evidence linked from affected completed tasks. Unavailable or skipped checks are never marked passed.
- [ ] Review implemented behavior against the complete architecture and document remaining nondeterminism or compatibility limitations. Acceptance: no required implementation work is hidden behind unchecked corpus-experiment results or unverified assumptions.

## Corpus Readiness and Experiment Gate

These items are separate from fixture-based implementation acceptance. Do not mark an actual experiment complete using synthetic results.

- [ ] Verify configured official gold splits, release/checksums, literature availability/source separation, and tokenizer compatibility against reconstructed training text. Acceptance: required inputs exist, originals remain immutable, and any source/segmentation limitations are recorded without tuning against test labels.
- [ ] Execute the frozen development search on real inputs. Acceptance: all planned trial outcomes, selected datasets/models, diagnostics, resource observations, and completion links are valid; actual fits versus reused results are distinguished. Leave test results unopened.
- [ ] Freeze the real experiment and perform final comparative evaluation. Acceptance: gate/final/control reports, paired bootstrap, signed differences, provenance, and completion checksums are valid; test results do not trigger further model selection. Reused baseline results are not reported as augmentation gains.

## Next Priorities and Blockers

- Implement M8 frozen comparative evaluation, paired bootstrap and standalone prediction.
- Establish official gold availability/release identity, literature source separation and tokenizer compatibility before real-corpus experiments.
- Preserve the restored lint/format baseline in subsequent work; see [refactoring verification](docs/HISTORY.md#refactoring-and-refinement). Historical verification results remain in [HISTORY.md](docs/HISTORY.md#verification-record).
