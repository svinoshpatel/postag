# Sprint Plan

See [AGENTS.md](AGENTS.md) for operating procedures and [docs/ARCH.md](docs/ARCH.md) for authoritative technical requirements. This plan tracks implementation and verification; it does not replace the architecture contracts. All implementation items remain unchecked until their acceptance criteria and applicable checks are satisfied.

## Sprint Goal and Current State

Implement a reproducible Ukrainian CRF self-training pipeline with isolated development and final evaluation, usable raw-text inference, and validated run artifacts. Preserve the approved experiment design and original datasets.

Architecture refinement is complete. Implementation status has not been established by this planning change; existing exploratory code is not evidence that the new contracts are satisfied. The repository still contains `postag/`, and the prescribed project configuration and `src/crf/` migration remain to be established.

Gold-corpus availability, release identity, and compatibility checks are pending. Small deterministic fixtures support implementation and offline verification without production data. Full corpus runs have a separate readiness gate below; missing inputs must not be replaced with another split or fabricated data. No sprint duration or delivery estimate is assumed.

## Ordered Implementation Milestones

Work in coherent increments. Before writing new tests for each increment, record the requirement-to-behavior mapping and boundary/failure cases, then resolve discrepancies. Run focused verification before dependent work and record results here. Independent implementation tasks may proceed once their prerequisites are met.

### 1. Establish the Environment and Core Package

Architecture: [Core Package and Migration Boundary](docs/ARCH.md#core-package-and-migration-boundary) and [Environment & Verification Contract](docs/ARCH.md).

- [ ] Establish the prescribed CPython/`uv` environment, dependency declarations, lockfile, interpreter selection, and Ruff/mypy/pytest configuration. Acceptance: dependency compatibility is demonstrated and all four verification tools are runnable from the repository root.
- [ ] Create installable `src/crf/` packaging and migrate or replace relevant `postag/` logic and affected tests/callers. Acceptance: pipeline imports use `crf`; reusable logic resides in the core, with no dependency on importing `src` as a package or preserving incompatible exploratory behavior.
- [ ] Add required generated-artifact exclusions before producing datasets, models, or reports. Acceptance: immutable inputs and existing artifacts are protected; intentional test fixtures remain identifiable.

### 2. Implement Shared Configuration, Artifact, and Resource Foundations

Prerequisite: milestone 1. Architecture: [Artifact & Run Contracts](docs/ARCH.md) and [CLI & Configuration Contract](docs/ARCH.md).

- [ ] Implement strict YAML configuration validation, documented defaults, seed precedence, version checks, and repository-relative path resolution. Acceptance: invalid/duplicate/unknown fields and incompatible settings fail explicitly; gold split aliases and source/output path violations are rejected.
- [ ] Implement typed sentence records, JSON/JSONL validators, canonical configuration hashing, deterministic sentence identities, and automatic input identities. Acceptance: exact schema fields, types, checksums, alignment, and nullability are enforced without exposing labels to feature code.
- [ ] Implement unique output allocation, publish-once completion handling, logging, and aggregate RSS monitoring. Acceptance: existing artifacts and raw inputs cannot be overwritten, diagnostics remain on stderr, budget violations fail, and incomplete runs cannot be treated as successful.

### 3. Implement Gold Parsing, Target Mapping, and Features

Prerequisites: milestones 1–2. Architecture: [Gold CoNLL-U Input Contract](docs/ARCH.md#gold-conll-u-input-contract), [Optional Target Tag Mapping](docs/ARCH.md#optional-target-tag-mapping), and [Ukrainian Language & NLP Feature Constraints](docs/ARCH.md).

- [ ] Implement the streaming CoNLL-U loader and separate label-free audit interface. Acceptance: supplied sentence order and integer-word boundaries are preserved; multiword/empty-node rows, malformed records, final delimiters, provenance, and diagnostics follow the parser contract.
- [ ] Implement optional service-tag mapping. Acceptance: original files and labels remain intact; source labels are validated first; all stages use the same configured target inventory; invalid mappings fail.
- [ ] Implement the shared versioned normalizer with source alignment. Acceptance: apostrophes, Unicode normalization, casing, whitespace, and Cyrillic preservation match the architecture, with original forms retained.
- [ ] Implement the feature dictionary schema and sentence-bounded context extraction. Acceptance: feature names/types, affix lengths, casing/shape flags, boundary flags, unseen forms, and annotation exclusion match the specification.

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

For the current plan refinement, architecture references, milestone ordering, acceptance criteria, and Markdown were checked by inspection. No executable behavior changed; implementation checks and corpus runs were not performed.

Update this table with the actual command result, date, and relevant evidence during implementation:

| Check | Command | Current result |
|---|---|---|
| Lint | `uv run ruff check .` | Not run for this documentation-only change |
| Formatting | `uv run ruff format --check .` | Not run for this documentation-only change |
| Types | `uv run mypy src/crf/` | Not run for this documentation-only change |
| Tests | `uv run pytest tests/ -v` | Not run for this documentation-only change |

## Open Dependencies and Blockers

- Environment/dependency compatibility and the package migration require implementation verification.
- Production corpus availability and tokenizer compatibility are unverified; the corpus gate cannot be completed until they are established.
- No implementation-check, training-run, or final-evaluation result is claimed by this plan update. Record discovered blockers, related failures, and their resolution here as work proceeds.
