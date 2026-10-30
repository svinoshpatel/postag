# Sprint Plan

See [AGENTS.md](AGENTS.md) for operating procedures and [docs/ARCH.md](docs/ARCH.md) for authoritative technical requirements. This plan tracks implementation and verification; it does not replace the architecture contracts. All implementation items remain unchecked until their acceptance criteria and applicable checks are satisfied.

## Sprint Goal and Current State

Implement a reproducible Ukrainian CRF self-training pipeline with isolated development and final evaluation, usable raw-text inference, and validated run artifacts. Preserve the approved experiment design and original datasets.

Milestone 1 is implemented: the locked CPython 3.14 environment and installable `src/crf/` foundations replace the incompatible `postag/` experiments. Milestone 2 is implemented: strict configuration, typed records and identities, safe artifact publication, logging, and resource monitoring are verified. Milestone 3 is implemented: gold parsing, shared target mapping, aligned normalization and strict sentence-bounded features are verified with offline fixtures. Later milestones remain unchecked. These components do not establish a complete raw-text preprocessing or training pipeline.

Gold-corpus availability, release identity, and compatibility checks are pending. Small deterministic fixtures support implementation and offline verification without production data. Full corpus runs have a separate readiness gate below; missing inputs must not be replaced with another split or fabricated data. No sprint duration or delivery estimate is assumed.

## Ordered Implementation Milestones

Work in coherent increments. Before writing new tests for each increment, record the requirement-to-behavior mapping and boundary/failure cases, then resolve discrepancies. Run focused verification before dependent work and record results here. Independent implementation tasks may proceed once their prerequisites are met.

### 1. Establish the Environment and Core Package

Architecture: [Core Package and Migration Boundary](docs/ARCH.md#core-package-and-migration-boundary) and [Environment & Verification Contract](docs/ARCH.md).

- [x] Establish the prescribed CPython/`uv` environment, dependency declarations, lockfile, interpreter selection, and Ruff/mypy/pytest configuration. Acceptance: dependency compatibility is demonstrated and all four verification tools are runnable from the repository root.
- [x] Create installable `src/crf/` packaging and migrate or replace relevant `postag/` logic and affected tests/callers. Acceptance: pipeline imports use `crf`; reusable logic resides in the core, with no dependency on importing `src` as a package or preserving incompatible exploratory behavior.
- [x] Add required generated-artifact exclusions before producing datasets, models, or reports. Acceptance: immutable inputs and existing artifacts are protected; intentional test fixtures remain identifiable.

Verified 2026-10-28: 20 tests and strict types (7 core files) pass; environment imports, packaging and tiny CRF persistence were checked. Review covered migration compatibility, surface-only features and tokenizer spans. Repository lint/format retain the unrelated failure recorded below; no production experiment was run.

#### PyCharm environment follow-up (2026-10-28)

- [x] Align project/module roots and inherited interpreter with the uv-managed Python 3.14 environment; preserve existing project-model changes.
- [x] Register the existing environment as a native uv SDK, add current preview/test run configurations, and document reload/setup behavior.
- [x] Validate XML/path references, interpreter/import compatibility, and all four required checks; record IDE validation limits and pre-existing failures.

Verified XML/paths, native uv SDK/imports and preview CLI; disabled the IDE module model that rewrote roots. SDK backup preserved; restart required to reload changes. UI Run/Debug/indexing remain unverified (connection timeouts). See [PyCharm setup](docs/PYCHARM.md).

### 2. Implement Shared Configuration, Artifact, and Resource Foundations

Prerequisite: milestone 1. Architecture: [Artifact & Run Contracts](docs/ARCH.md) and [CLI & Configuration Contract](docs/ARCH.md).

- [x] Implement strict YAML configuration validation, documented defaults, seed precedence, version checks, and repository-relative path resolution. Acceptance: invalid/duplicate/unknown fields and incompatible settings fail explicitly; gold split aliases and source/output path violations are rejected.
- [x] Implement typed sentence records, JSON/JSONL validators, canonical configuration hashing, deterministic sentence identities, and automatic input identities. Acceptance: exact schema fields, types, checksums, alignment, and nullability are enforced without exposing labels to feature code.
- [x] Implement unique output allocation, publish-once completion handling, logging, and aggregate RSS monitoring. Acceptance: existing artifacts and raw inputs cannot be overwritten, diagnostics remain on stderr, budget violations fail, and incomplete runs cannot be treated as successful.

Completed 2026-10-28 with temporary offline fixtures; changed-code lint/format passed, with the repository-wide exception below. Reviews/tests covered strict configuration/path boundaries, schema/identity/hash integrity, no-clobber lifecycle races and sticky RSS failures. Stage-specific validators and pipeline integration remain later work.

| Task | New tests | Full suite | Typed core files |
|---|---:|---:|---:|
| Configuration/paths | 88 | 108 | 9 |
| Records/serialization/identity | 119 | 227 | 12 |
| Lifecycle/logging/resources | 47 | 274 | 15 |

### 3. Implement Gold Parsing, Target Mapping, and Features

Prerequisites: milestones 1–2. Architecture: [Gold CoNLL-U Input Contract](docs/ARCH.md#gold-conll-u-input-contract), [Optional Target Tag Mapping](docs/ARCH.md#optional-target-tag-mapping), and [Ukrainian Language & NLP Feature Constraints](docs/ARCH.md).

- [x] Implement the streaming CoNLL-U loader and separate label-free audit interface. Acceptance: supplied sentence order and integer-word boundaries are preserved; multiword/empty-node rows, malformed records, final delimiters, provenance, and diagnostics follow the parser contract.
- [x] Implement optional service-tag mapping. Acceptance: original files and labels remain intact; source labels are validated first; all stages use the same configured target inventory; invalid mappings fail.
- [x] Implement the shared versioned normalizer with source alignment. Acceptance: apostrophes, Unicode normalization, casing, whitespace, and Cyrillic preservation match the architecture, with original forms retained.
- [x] Implement the feature dictionary schema and sentence-bounded context extraction. Acceptance: feature names/types, affix lengths, casing/shape flags, boundary flags, unseen forms, and annotation exclusion match the specification.

Completed with offline fixtures; changed-code lint/format passed at every step, with the repository-wide exception below. Pre-test reviews covered parser/audit isolation, source/target inventory consistency, normalization/alignment and exact feature/context schemas. Compatibility boundaries and remaining integration work are specified in ARCH.md.

| Task | Date | New tests | Full suite | Typed core files |
|---|---|---:|---:|---:|
| Gold parser/audit | 2026-10-28 | 59 | 333 | 16 |
| Shared tag mapping | 2026-10-28 | 89 | 422 | 17 |
| Aligned normalization/record conversion | 2026-10-28 | 73 | 495 | 18 |
| Feature schema/context | 2026-10-29 | 99 | 594 | 19 |

M3 is complete for fixture-based acceptance; real-corpus readiness, the production raw-text/UD adapter and model/orchestration stages remain pending. Original inputs/artifacts and unrelated `main.py` state were preserved; no real-corpus training/evaluation is claimed.

### 4. Implement Literature Streaming and Contamination Controls

Prerequisites: milestones 2–3. Architecture: [Raw-Text Segmentation Contract](docs/ARCH.md#raw-text-segmentation-contract) and [Strict Test Data Isolation & Leakage Prevention](docs/ARCH.md).

- [ ] Implement flat literature discovery, source encoding, and the versioned tokenizer/UD adapter. Acceptance: discovery never recurses or consumes gold; paragraphs and sentences survive buffer boundaries; original offsets and configured size limits are respected.
- [ ] Implement text-only gold overlap exclusion, deterministic literature deduplication, and known-source/MWT exclusions. Acceptance: automatic fingerprints and exclusion diagnostics follow the contract; held-out labels remain uninterpreted; original gold is unchanged.
- [ ] Verify streaming and resource-failure behavior with bounded fixtures and controlled monitor inputs. Acceptance: oversized candidates are rejected intact, gold sentences are not filtered or truncated, intermediates are incremental, and inference-only filtering is distinct from pseudo-label selection.

### 5. Implement CRF Training, Model Bundles, and Validation Metrics

Prerequisites: milestones 2–3. Architecture: [Training & Model-Selection Contract](docs/ARCH.md), [Model Bundles](docs/ARCH.md#model-bundles), and [Metric Records](docs/ARCH.md#metric-records).

- [ ] Implement the CRF training wrapper with the prescribed optimizer settings, sequential fitting, termination records, and memory failures. Acceptance: every candidate uses only its declared training records; iteration-limited models are not called converged.
- [ ] Implement model serialization, metadata/checksum validation, and save/load verification. Acceptance: round-tripped predictions and confidences agree, mismatched bundles fail, and standalone loading does not require original training files.
- [ ] Implement validation metrics and reports. Acceptance: token accuracy, fixed-inventory macro/per-tag F1, confusion counts, shared gold-only OOV vocabulary, and punctuation diagnostics agree with deterministic fixtures, including empty diagnostic subsets.
- [ ] Implement the gold-only candidate search and gate selection. Acceptance: the nine baseline configurations, validation-only selection, tie breakers, cached candidates, and trial records match the contract; test predictions and metrics are unavailable during development.

### 6. Implement Pseudo-Label Confidence and Sampling

Prerequisites: milestones 4–5. Architecture: [Confidence & Sequence Acceptance](docs/ARCH.md) and [Pseudo-Label Sampling Contract](docs/ARCH.md#pseudo-label-sampling-contract).

- [ ] Implement Viterbi-label marginals and complete-sentence acceptance. Acceptance: no token deletion or maximum-marginal substitution occurs; unrounded confidences and the minimum-grid-threshold candidate pool support every configured threshold.
- [ ] Implement deterministic disk-backed priority ordering and whole-sentence sampling. Acceptance: the approved token cap is respected, oversized remaining-budget candidates are skipped without splitting, underfilled selections are valid, and each threshold's frozen dataset is reused across its fits.
- [ ] Record acceptance, exclusion, source/POS distribution, and cap-utilization diagnostics. Acceptance: empty selections reuse corresponding baseline candidates and cached validation results without retraining or claiming augmentation gains.

### 7. Orchestrate Development and Freeze Selection

Prerequisites: milestones 2–6. Architecture: [Frozen Model Selection](docs/ARCH.md#frozen-model-selection), [Automatic Run and Prediction Records](docs/ARCH.md#automatic-run-and-prediction-records), and [CLI & Configuration Contract](docs/ARCH.md).

- [ ] Implement the `develop` CLI and final-model search. Acceptance: the default experiment accounts for nine baseline plus 27 final configurations; each final fit starts fresh unless baseline reuse applies; failed/unexecuted trials remain visible rather than shrinking the search.
- [ ] Implement final selection, matched gold-only control, immutable run history, and frozen model references. Acceptance: selection uses the prescribed validation tie breakers, all linked identities/checksums agree, and only a complete successful development run authorizes final evaluation.

### 8. Implement Frozen Evaluation and Standalone Prediction

Prerequisites: milestones 5–7 for final evaluation; milestones 2–5 for standalone prediction. Architecture: [Evaluation Contract](docs/ARCH.md#evaluation-contract), [Final Evaluation Reports](docs/ARCH.md#final-evaluation-reports), and [Standalone Prediction Reports](docs/ARCH.md#standalone-prediction-reports).

- [ ] Implement the `evaluate` CLI, three-model alignment, comparisons, and test reports. Acceptance: no refitting or overrides occur; changed/incomplete experiments fail; valid completed results are returned from cache; development history remains unchanged.
- [ ] Implement paired bootstrap and persisted resampling units. Acceptance: grouping/fallback, random generator, repeated-unit weighting, pooled accuracy, percentile interpolation, unavailable intervals, and integrity links follow the contract.
- [ ] Implement the `predict` CLI and fresh standalone reports. Acceptance: recorded model preprocessing is used, low-confidence and repeated sentences remain eligible, resource exclusions are reported, and custom mapped labels are not presented as standard UD UPOS.

### 9. Verify Integration and Complete Implementation

Prerequisites: milestones 1–8. Architecture: [Technical Verification Invariants](docs/ARCH.md#technical-verification-invariants); operating procedure: [Verification and Test Generation](AGENTS.md#4-verification-and-test-generation).

- [ ] Add and run deterministic integration coverage using small, intentional fixtures. Acceptance: the full develop/freeze/evaluate/predict flow, default search accounting, empty augmentation, isolation boundaries, schema/checksum rejection, and incomplete-output behavior are verified without production data or network access. Integration dependencies are documented.
- [ ] Run all four required checks and record actual results below. Acceptance: changed behavior is verified; any permitted unrelated pre-existing failures have evidence and are recorded beside affected completed tasks. Unavailable or skipped checks are never marked passed.
- [ ] Review implemented behavior against the complete architecture and document remaining nondeterminism or compatibility limitations. Acceptance: no required implementation work is hidden behind unchecked corpus-experiment results or unverified assumptions.

## Corpus Readiness and Experiment Gate

These items are separate from fixture-based implementation acceptance. Do not mark an actual experiment complete using synthetic results.

- [ ] Verify configured official gold splits, release/checksums, literature availability/source separation, and tokenizer compatibility against reconstructed training text. Acceptance: required inputs exist, originals remain immutable, and any source/segmentation limitations are recorded without tuning against test labels.
- [ ] Execute the frozen development search on real inputs. Acceptance: all planned trial outcomes, selected datasets/models, diagnostics, resource observations, and completion links are valid; actual fits versus reused results are distinguished. Leave test results unopened.
- [ ] Freeze the real experiment and perform final comparative evaluation. Acceptance: gate/final/control reports, paired bootstrap, signed differences, provenance, and completion checksums are valid; test results do not trigger further model selection. Reused baseline results are not reported as augmentation gains.

## Verification Record

Latest code checks: 2026-10-29, CPython 3.14.6/macOS arm64. Historical results below were not rerun for the documentation compaction on 2026-10-30; references, task state and scoped diffs were checked.

| Check | Command | Current result |
|---|---|---|
| Lint | `uv run ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001; changed-code lint passed |
| Formatting | `uv run ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py` requires formatting; changed-code formatting passed |
| Types | `uv run mypy src/crf/` | Passed (2026-10-29): 19 source files |
| Tests | `uv run pytest tests/ -v` | Passed (2026-10-29): 594 tests, including 110 feature tests, 73 normalization/record cases, 89 shared label-space tests, 59 gold parser/audit tests and two offline integration tests |

## Open Dependencies and Blockers

- Runtime imports, `uv lock --check`/`uv pip check`, isolated editable imports, wheel/sdist contents and artifact ignore rules were verified during M1–M2. Tiny offline CRF fitting/marginal/joblib checks passed; cross-platform bitwise reproducibility is not claimed.
- Lint/format exception: unchanged `scripts/fetch_dictionary.py` fails I001 and formatting, reproduced before/after implementation. Changed-code checks pass; no checks were weakened. Existing `AD main.py` state remains untouched.
- Official gold availability/release identity, source separation and tokenizer compatibility remain unverified. M4 must handle upstream standalone formatting-character omissions, UD boundaries and aligned normalization with bounded paragraph streaming. No production training or final evaluation has run.
