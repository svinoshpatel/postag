# Sprint Plan

See [AGENTS.md](AGENTS.md) for operating procedures and [docs/ARCH.md](docs/ARCH.md) for authoritative technical requirements. This plan tracks implementation and verification; it does not replace the architecture contracts. All implementation items remain unchecked until their acceptance criteria and applicable checks are satisfied.

## Sprint Goal and Current State

Implement a reproducible Ukrainian CRF self-training pipeline with isolated development and final evaluation, usable raw-text inference, and validated run artifacts. Preserve the approved experiment design and original datasets.

Milestone 1 is implemented: the locked CPython 3.14 environment and installable `src/crf/` foundations replace the incompatible `postag/` experiments. Milestone 2 is implemented: strict configuration, typed records and identities, safe artifact publication, logging, and resource monitoring are verified. Milestone 3 gold parsing, shared target mapping and aligned normalization are implemented; feature-contract completion remains pending. Later milestones remain unchecked. The tokenizer backend and surface features do not establish a complete preprocessing or training pipeline.

Gold-corpus availability, release identity, and compatibility checks are pending. Small deterministic fixtures support implementation and offline verification without production data. Full corpus runs have a separate readiness gate below; missing inputs must not be replaced with another split or fabricated data. No sprint duration or delivery estimate is assumed.

## Ordered Implementation Milestones

Work in coherent increments. Before writing new tests for each increment, record the requirement-to-behavior mapping and boundary/failure cases, then resolve discrepancies. Run focused verification before dependent work and record results here. Independent implementation tasks may proceed once their prerequisites are met.

### 1. Establish the Environment and Core Package

Architecture: [Core Package and Migration Boundary](docs/ARCH.md#core-package-and-migration-boundary) and [Environment & Verification Contract](docs/ARCH.md).

- [x] Establish the prescribed CPython/`uv` environment, dependency declarations, lockfile, interpreter selection, and Ruff/mypy/pytest configuration. Acceptance: dependency compatibility is demonstrated and all four verification tools are runnable from the repository root.
- [x] Create installable `src/crf/` packaging and migrate or replace relevant `postag/` logic and affected tests/callers. Acceptance: pipeline imports use `crf`; reusable logic resides in the core, with no dependency on importing `src` as a package or preserving incompatible exploratory behavior.
- [x] Add required generated-artifact exclusions before producing datasets, models, or reports. Acceptance: immutable inputs and existing artifacts are protected; intentional test fixtures remain identifiable.

Milestone 1 verification exception: repository-wide Ruff lint and formatting fail only on the unchanged `scripts/fetch_dictionary.py` (import ordering and formatting), reproduced before migration. All changed Python files pass both checks; strict core types and all 20 tests pass. See the verification record below.

#### Milestone 1 execution steps (2026-10-28)

1. Establish CPython 3.14, declare runtime/dev dependencies, lock and sync with `uv`, and verify tool availability before migrating code.
2. Replace incompatible exploratory features and the spaCy tokenizer with typed `crf` foundations; update callers, fixtures, and usage documentation. Review requirements/boundaries before writing replacement tests and run focused checks.
3. Exclude generated artifacts, verify ignore rules without creating artifacts, and run all four required checks. Record environment and migration limitations.

#### Milestone 1 specification review before replacement tests

- Packaging/runtime: Hatch installs only `src/crf`; `.python-version` selects CPython 3.14.6, and the lock resolves runtime dependencies and all three check tools. All imports and tool availability were verified through `uv`; initial core mypy passed.
- Migration: no new core imports `postag` or `src`. Retired TF/SVD and spaCy whole-file behavior is deliberately incompatible, as documented before implementation in the architecture. Features accept surface forms only, emit finite typed values, use ±2 sentence context, omit unavailable short affixes, and retain unseen strings without learned statistics.
- Text backend: immutable paragraph-relative spans recover original apostrophes from offsets. Empty/whitespace inputs yield no sentences. Full normalization/alignment, format-character handling, UD adaptation and bounded file reading remain later tasks; this preview cannot produce pipeline artifacts.
- Replacement tests will derive feature values and boundaries from the architecture, check invalid/annotated inputs, original code-point slices and immutable records, and exercise installed-package imports and compiled CRFsuite fitting/serialization with tiny deterministic fixtures. The latter checks dependency compatibility, not experiment quality.
- Baseline checks found legacy lint/format failures, including the unrelated `scripts/fetch_dictionary.py`. Preserve that file and record its failure if it remains after migration; do not weaken repository checks.

#### PyCharm environment follow-up (2026-10-28)

- [x] Align project/module roots and inherited interpreter with the uv-managed Python 3.14 environment; preserve existing project-model changes.
- [x] Register the existing environment as a native uv SDK, add current preview/test run configurations, and document reload/setup behavior.
- [x] Validate XML/path references, interpreter/import compatibility, and all four required checks; record IDE validation limits and pre-existing failures.

PyCharm follow-up verification: parsed all project XML, resolved source/test roots to existing directories, checked the inherited SDK against the local interpreter registry (Python 3.14.6, native uv, pip disabled), imported `crf` through `.venv/bin/python`, and executed the shared preview command successfully. A backup of the prior local SDK registry was saved before the targeted update; other SDK entries were preserved. The automatic pyproject module model was observed rewriting roots incorrectly and is disabled as documented in `docs/ARCH.md` and `docs/PYCHARM.md`. PyCharm must restart to reload the externally updated SDK registry. UI verification was unavailable because both native-app connection attempts timed out; no successful IDE Run/Debug or indexing result is claimed. Full checks rerun: mypy passed (7 source files), pytest passed (20 tests); lint/format retain only the previously documented unchanged `scripts/fetch_dictionary.py` failures. Concurrent IDE changes to `vcs.xml`/source-root detection and pre-existing Git changes were not staged or committed by this task.

### 2. Implement Shared Configuration, Artifact, and Resource Foundations

Prerequisite: milestone 1. Architecture: [Artifact & Run Contracts](docs/ARCH.md) and [CLI & Configuration Contract](docs/ARCH.md).

- [x] Implement strict YAML configuration validation, documented defaults, seed precedence, version checks, and repository-relative path resolution. Acceptance: invalid/duplicate/unknown fields and incompatible settings fail explicitly; gold split aliases and source/output path violations are rejected.
- [x] Implement typed sentence records, JSON/JSONL validators, canonical configuration hashing, deterministic sentence identities, and automatic input identities. Acceptance: exact schema fields, types, checksums, alignment, and nullability are enforced without exposing labels to feature code.
- [x] Implement unique output allocation, publish-once completion handling, logging, and aggregate RSS monitoring. Acceptance: existing artifacts and raw inputs cannot be overwritten, diagnostics remain on stderr, budget violations fail, and incomplete runs cannot be treated as successful.

Milestone 2 task 1 verification (2026-10-28): 88 new deterministic configuration/path tests pass; all 108 repository tests and strict core types (9 source files) pass. Changed-code Ruff lint and formatting pass. Full repository lint/format retain only the unchanged `scripts/fetch_dictionary.py` failures, reproduced before implementation (I001 and required formatting); this is the permitted pre-existing-failure exception. No checks were weakened, no production inputs were read, and no pipeline artifacts were generated. Unique output allocation, content auditing, artifact schemas/hashes, and future pipeline CLIs remain unchecked subsequent tasks.

#### Milestone 2, task 1 execution steps (2026-10-28)

1. Define the configuration API and document unresolved schema/path/domain details before implementation.
2. Implement strict safe YAML loading, immutable effective settings, seed precedence, and read-only path validation in `src/crf/utils/`.
3. Record specification review and boundary cases before adding deterministic offline tests; run focused tests and all four required checks.

#### Milestone 2, task 1 specification review before tests

- Safe YAML: a local SafeLoader constructor rejects duplicate/non-string keys at every depth, unsafe tags, merge keys, and recursive aliases. Each section enforces an exact field allowlist, required fields, and strict types; booleans cannot pass as numbers. Nonfinite values, unsupported versions, invalid encodings, empty/duplicate grids, invalid tag consolidation, and resource incompatibility fail.
- Immutable records contain all defaults; only an explicit CLI seed overrides YAML, while the bootstrap seed stays independent. Effective dictionaries contain POSIX corpus paths and JSON-compatible arrays/numbers. Zero regularization, thresholds at 0/1, ratio zero, and unlimited CRF iterations are valid boundaries.
- Path checks resolve relative CLI/config/data paths from the explicit repository root. Gold files must exist, remain under gold, and have distinct device/inode identities; literature uses its prescribed directory without symlinks. Validation reads no annotation contents. Metadata-only parsing does not require original files.
- Output checks are read-only and reject existing/dangling entries, symlink ancestors, raw destinations, and artifact-tree escapes. Atomic allocation/publication, corpus discovery/content auditing, hashing and artifact schemas remain subsequent tasks.
- Tests will use temporary repositories and opaque gold-file bytes, exercise all sections and defaults/overrides, malformed YAML and nested duplicates, hard/symbolic aliases, missing/escaping sources, and fresh versus existing/redirected output paths. No production input or generated artifact is touched.

Milestone 2 task 2 verification (2026-10-28): 119 new offline serialization/identity/record tests pass; the full suite passes 227 tests, and strict mypy passes all 12 core files. Changed-code Ruff lint and formatting pass. Repository-wide Ruff lint/format still fail only on unchanged `scripts/fetch_dictionary.py` (I001/formatting), reproduced before implementation; this is the permitted pre-existing-failure exception. All checks use `uv`; no checks were weakened. Tests cover opaque undecodable gold bytes, exact Unicode JSON/hash encodings, strict nested schemas, confidence endpoints/full precision, mapped inventories, global spans/null spans, test-label opt-in, source changes, and path/source isolation. Production sources and existing outputs were untouched; the unrelated `AD main.py` state remains unchanged. Fingerprint/model-content verification and shared-normalizer semantics remain later-stage requirements, as documented in the architecture; output allocation/publication remains milestone 2 task 3.

#### Milestone 2, task 2 execution steps (2026-10-28)

1. Define strict JSON helpers, immutable dataset/input records, and validation context; document the boundary with later loaders, normalization, and artifact publication.
2. Implement canonical configuration bytes/checksums, provenance-derived sentence identities, bounded opaque-byte input hashing, and strict streaming record validation.
3. Review requirements and failure boundaries before writing tests; verify deterministic fixtures, full checks, and preservation of raw inputs and existing artifacts.

#### Milestone 2, task 2 specification review before tests

- Strict JSON helpers reject nested duplicate keys, nonfinite literals/overflow, BOMs, invalid Unicode, non-plain encoding inputs, blank JSONL records, and non-LF line endings. JSONL iteration validates one object at a time with path/line diagnostics; encoding helpers return bytes without publishing files.
- Sentence factories enforce exact top-level/nested fields, schema version 1, strict integer/probability types, nonempty aligned arrays, consecutive IDs, ordered bounded spans (requiring decoded-source length), gold/pseudo nullability, original UD inventory, configured target mapping/inventory, input bindings, provenance IDs, minimum confidence, and seed-derived priority. Test-label records require explicit final evaluation. Immutable tuples preserve alignment; features receive forms only.
- Canonical effective-config hashing uses complete settings and the exact sorted compact UTF-8 JSON encoding without newline. Sentence IDs use the specified ordered provenance array. File checksums/size use original bounded byte reads and observed-change detection; identity verification rehashes current sources. Input discovery is flat and deterministic, rejects symlinks/gold aliases and empty literature, and does not interpret annotations.
- Deterministic tests will independently derive hash expectations, round-trip gold/pseudo/input records, mutate schema/types/alignments/nullability/provenance/spans, exercise strict JSON failure diagnostics and incremental reading, and verify source order, opaque test bytes, path failures and changed-byte rejection. Fingerprint/model-content and linguistic normalization verification remain with their later owning stages; no production data, model artifact, or output publication is needed.

Milestone 2 completion verification (2026-10-28): all three foundation tasks meet their fixture-based acceptance criteria. Task 3 adds 47 passing lifecycle/resource tests, including an isolated child-process interruption integration test; the full suite passes 274 tests and strict mypy passes all 15 core files. New Python files pass separate Ruff lint and formatting checks. Repository-wide lint and formatting retain only the unchanged `scripts/fetch_dictionary.py` I001/formatting failures, reproduced before implementation; this is the permitted pre-existing-failure exception. Checks were not weakened. `git diff --check` passes. Raw data and existing generated artifacts were untouched; all new lifecycle artifacts were created only in pytest temporary directories, covered by existing ignore patterns when used in the repository. The pre-existing staged/worktree state of `main.py` was preserved.

Lifecycle limits: POSIX directory descriptors/no-follow traversal and hard-link publication are verified on macOS; unsupported filesystems fail rather than falling back to clobbering writes. RSS monitoring is best effort, with documented sampling/native-code limits and caller-owned worker cleanup. Later orchestration stages must supply their complete schema/link validators and integrate the monitor; no development search, model training run, final evaluation, or corpus-readiness result is claimed by this milestone.

#### Milestone 2, task 3 execution steps (2026-10-28)

1. Document run-allocation, atomic publication, completion-validation, logging, and memory-monitor integration boundaries before implementation.
2. Implement reusable lifecycle helpers with no-clobber filesystem operations, explicit completion validators, stderr diagnostics, and sticky aggregate-RSS failures.
3. Record specification/boundary review before deterministic failure/race tests; run focused and full required checks, then mark milestone 2 complete only when acceptance is verified.

#### Milestone 2, task 3 specification review before tests

- Allocation uses UUID run IDs and exclusive directory creation, checks every artifact tree for collisions, and never reuses partial runs. Development creates four trees; prediction creates reports only. Resolved path checks and no-follow descriptor traversal protect raw inputs and reject symlinked ancestors. Partial allocation fails without a completion record.
- Atomic writers stage new temporary entries, flush/fsync, validate a read-only stream, and hard-link the final name without replacement. Body/validation failures leave no final name; competing or pre-existing files/directories/dangling links survive. Cleanup removes only this writer's temporary entry.
- Completion helpers validate an encoded snapshot, containing run identity, version/status/error semantics, exact monitor peak and finished monitoring. Successful publication additionally requires no latched monitoring failure or context-body abort; failed records can still describe failure. A mandatory caller validator owns exact stage schemas and all linked integrity checks. Loading rejects absent/failed/partial/invalid descriptors and never infers success from files/directories alone.
- Logging installs one stderr handler for the crf hierarchy without root-logger changes. The mutable RSS controller samples root plus recursive workers, tolerates exited children, fails closed on monitoring errors, permits budget equality, retains the peak/first failure, interrupts the main thread on background failure, and raises at checkpoints/context exit. Later pipeline stages must integrate it and clean up owned workers; sampling/native-code limits are documented.
- Tests will use temporary artifact trees and controlled samplers/process observations, cover collisions and concurrent publication, validation/write failures, existing/raw/symlink protection, missing/failed/corrupt completions, checksum-validator rejection, logging isolation, root/worker RSS accounting, sticky failures, interruption and clean monitor shutdown. No production dataset or existing generated artifact will be changed.

### 3. Implement Gold Parsing, Target Mapping, and Features

Prerequisites: milestones 1–2. Architecture: [Gold CoNLL-U Input Contract](docs/ARCH.md#gold-conll-u-input-contract), [Optional Target Tag Mapping](docs/ARCH.md#optional-target-tag-mapping), and [Ukrainian Language & NLP Feature Constraints](docs/ARCH.md).

- [x] Implement the streaming CoNLL-U loader and separate label-free audit interface. Acceptance: supplied sentence order and integer-word boundaries are preserved; multiword/empty-node rows, malformed records, final delimiters, provenance, and diagnostics follow the parser contract. Verified with 59 offline parser tests; full checks retain only the documented, unrelated `scripts/fetch_dictionary.py` lint/format failures.
- [x] Implement optional service-tag mapping. Acceptance: original files and labels remain intact; source labels are validated first; all stages use the same configured target inventory; invalid mappings fail. Shared API verified across existing configuration/parser/record stages with 89 new offline tests; future stages must consume that API. Full checks retain only the documented unrelated lint/format failures.
- [x] Implement the shared versioned normalizer with source alignment. Acceptance: apostrophes, Unicode normalization, casing, whitespace, and Cyrillic preservation match the architecture, with original forms retained. Verified with 73 new normalization/record tests; full checks retain only the documented unrelated lint/format failures.
- [ ] Implement the feature dictionary schema and sentence-bounded context extraction. Acceptance: feature names/types, affix lengths, casing/shape flags, boundary flags, unseen forms, and annotation exclusion match the specification.

#### Milestone 3, task 1 execution steps (2026-10-28)

1. Define immutable parsed-gold and label-free audit interfaces, reconstruction rules, and the boundary with later mapping/normalization.
2. Implement strict incremental CoNLL-U framing, word/auxiliary-row validation, provenance, and surface reconstruction without interpreting held-out annotations in audits.
3. Record specification review before tests; verify deterministic normal/boundary/failure fixtures and all four required checks. Preserve existing Git state and production inputs.

#### Milestone 3, task 1 specification review before tests

- Incremental binary-line framing decodes strict UTF-8 with physical-line diagnostics, accepts LF/CRLF, requires final blank delimiters, and rejects comments after rows, malformed fields, BOMs, and sentences without words. No gold size/token limits or silent repairs are applied.
- Immutable parser outputs preserve consecutive integer IDs, forms (including literal underscores), original UPOS, comments, sentence order, source identity and IDs. Targets use the already-validated configuration mapping after source UPOS validation. Structural range/decimal rows stay out of labels and word counts; reconstruction retains MWT surfaces and spacing while excluding empty nodes. Missing range members, overlap, duplicates, invalid IDs/UPOS, and observed source changes fail.
- Audits interpret only the whitelisted surface/document metadata, IDs, FORM and spacing; annotation fields remain opaque and never enter returned objects. Test-label loading rejects absent or nonboolean evaluation opt-in before reading sources. Features/normalization, mapped artifact construction, contamination filtering and full pipeline orchestration remain subsequent tasks.
- Tests will independently check provenance hashes, original/reconstructed surfaces, document continuity, annotation-poisoned audits, strict failure path/line diagnostics, incremental consumption, unchanged source bytes, and large intact gold sentences using only temporary offline fixtures.

Milestone 3 task 1 verification (2026-10-28): all 59 new parser tests pass; the full suite passes 333 tests and strict mypy passes all 16 core source files. Changed-file Ruff lint and formatting pass separately. Repository-wide lint/format still fail only on unchanged `scripts/fetch_dictionary.py` (I001 and formatting), reproduced before implementation and after the final code changes; the permitted pre-existing-failure exception applies. No checks were weakened. `git diff --check` passes. Tests cover exact provenance, label-free poisoned-annotation audits, explicit test-label access, reconstruction/MWT/empty nodes, strict diagnostics, streaming, source changes and gold sentences exceeding raw-text limits. Production corpus files and generated artifacts were untouched, and the pre-existing `AD main.py` state was preserved.

At task 1 completion, normalized dataset assembly, broader target-mapping integration, contamination controls and real-corpus readiness remained later tasks. Source validation requires iterator exhaustion; a late failure invalidates the source even if prior sentences were yielded. Later pipeline consumers must integrate memory monitoring and atomic artifact publication.

#### Milestone 3, task 2 execution steps (2026-10-28)

1. Define one immutable label-space API and document integration/compatibility boundaries; preserve configuration and dataset wire formats.
2. Centralize source UPOS validation, mapping validation/application and sorted target inventory; integrate configuration, gold parsing and gold/pseudo record validation.
3. Record specification review before new tests; verify all splits, preservation, invalid mappings, target-space mismatches and serialization using offline fixtures, then run all four required checks.

#### Milestone 3, task 2 specification review before tests

- `TagMapping` validates immutable entries (including duplicate/malformed direct inputs), canonicalizes order, accepts only the complete four-service-tag consolidation or disabled mapping, and rejects invalid/colliding/non-UTF-8 targets. Mapping and target checks return immutable aligned tuples; inputs remain untouched. Inventories derive from all 17 source tags and are alphabetically sorted (17 default, 14 consolidated).
- Configuration parsing/serialization, labeled loading and gold/pseudo record validation share the API; configuration constants remain compatible re-exports. Source UPOS validation precedes mapping and target comparisons. Pseudo labels are validated directly without renaming or confidence aggregation. Direct configuration misuse fails at label-space consumers too.
- Audit reading remains independent of mapping and label interpretation. Train/validation records and explicitly authorized test records must agree on targets; mapping changes alter configuration identity and invalidate mismatched records. Existing schema fields/versions remain unchanged; normalized artifact assembly and future model/metric/inference integration remain with their owning tasks.
- Existing 212 configuration/parser/record tests and strict core types pass after integration. New tests will independently specify fixed inventories and outputs; cover all tags/splits, invalid mappings/types/Unicode/duplicates, immutable preservation, poisoned audit inputs, JSON/config round trips and changed-mapping rejection using temporary fixtures only.

Milestone 3 task 2 verification (2026-10-28): 89 new offline label-space tests pass, and the full suite passes 422 tests. Strict mypy passes all 17 core source files; changed-file Ruff lint and formatting pass separately. Repository-wide lint/format fail only on unchanged `scripts/fetch_dictionary.py` (I001 and formatting), reproduced before and after implementation; the permitted pre-existing-failure exception applies. `git diff --check` passes. No checks were weakened. Fixtures verify every UD tag, 17/14-class inventories, exact custom Unicode targets, mapping preservation/config identity, source/target validation order, immutable records and JSON round trips across train/validation/explicit final-test loading, unmapped pseudo rejection without confidence changes, and label-free audit independence. Production inputs/generated artifacts and the pre-existing `AD main.py` state remain untouched.

At task 2 completion, mapping was centralized for all implemented consumers, with compatible configuration and dataset wire formats. Future model/metric/prediction stages must use the shared label space and their prescribed inventory metadata. At that point, those stages, normalized dataset assembly and production-corpus readiness remained unimplemented. No post-hoc mapped model/confidence conversion is provided or claimed.

#### Milestone 3, task 3 execution steps (2026-10-28)

1. Specify versioned text/form normalization, conservative source-span projection and formatting diagnostics, including the boundary with future paragraph streaming/tokenization.
2. Implement the shared normalizer; integrate existing features, dataset form validation and gold-record conversion without changing originals or artifact schemas.
3. Record specification/boundary review before tests; verify transformations, Unicode alignment/failures, source preservation and existing consumers with offline fixtures, then run all four required checks.

#### Milestone 3, task 3 specification review before tests

- Raw text normalizes a single file-start BOM, CRLF/CR, NFC, apostrophes and NBSP variants in the required order, preserving case, line/paragraph boundaries and other characters. Form normalization preserves leading BOMs and supplied token boundaries; lexical lowercasing is separate and NFC is applied again afterward. Version/Unicode metadata and strict scalar/type validation are explicit.
- Immutable alignment records assign each output character its entire original NFC-unit interval, including interacting starter compositions, canonical reordering and expansions. Absolute offsets count original decoded code points, CRLF spans both characters, and projection rejects partial-unit/empty/reversed/out-of-bounds spans. Complete-unit projection must recover an original slice that normalizes to exactly the selected output. No chunk-based/file streaming or raw filtering is introduced.
- Formatting characters remain visible in normalized text except one permitted raw file-start BOM. Structured observations retain every original offset/removal flag, with module-level logging summaries. Stress marks, soft hyphens, zero-width characters, Cyrillic distinctions and Latin lookalikes remain preserved under NFC.
- Features re-export/use the shared form normalizer; record validation now checks exact normalized-form semantics. Gold conversion preserves originals, labels, provenance, null word spans and separate parser metadata; mapped targets are revalidated and test records retain explicit opt-in. Existing 224 affected tests and strict types pass. New deterministic tests will derive outputs/spans independently, exercise multiple scripts, projection boundaries, failure cases, log diagnostics, record conversion/round trips and source immutability.

Milestone 3 task 3 verification (2026-10-28): 73 new offline normalization/record cases pass; the full suite passes 495 tests and strict mypy passes all 18 core source files. Changed-file Ruff lint and formatting pass separately. Repository-wide lint/format retain only unchanged `scripts/fetch_dictionary.py` (I001 and formatting), reproduced before and after implementation; the permitted pre-existing-failure exception applies. `git diff --check` passes. No checks were weakened. Coverage includes exact raw offsets/BOM/CRLF, NFC composition/expansion/reordering across scripts, 80 bounded fixed-seed Unicode reference cases, complete-unit projection and ambiguous cuts, Cyrillic/case/stress/format-character preservation and diagnostics, invalid text/version/span inputs, mapped gold-record conversion for all splits with explicit test access, source immutability, and gold/pseudo normalized-form rejection.

Normalization version 1 and Unicode-version metadata are available through immutable text results. Conservative alignment rejects partial NFC-unit spans instead of inventing character offsets. The raw-text/UD streaming adapter, contamination fingerprints/exclusions, production tokenizer compatibility, model stages and corpus readiness remain later work. Existing low-level tokenizer/preview behavior is preserved. Original datasets and existing generated artifacts were untouched; the unrelated `AD main.py` state remains unchanged.

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

Milestone 1 verification performed on 2026-10-28 with CPython 3.14.6 on macOS arm64. Architecture references, migrated API documentation, and command examples were inspected. No production corpus processing or experiment was performed.

Update this table with the actual command result, date, and relevant evidence during implementation:

| Check | Command | Current result |
|---|---|---|
| Lint | `uv run ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001; changed-code lint passed |
| Formatting | `uv run ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py` requires formatting; changed-code formatting passed |
| Types | `uv run mypy src/crf/` | Passed (2026-10-28): 18 source files |
| Tests | `uv run pytest tests/ -v` | Passed (2026-10-28): 495 tests, including 73 normalization/record cases, 89 shared label-space tests, 59 gold parser/audit tests and two offline integration tests |

## Open Dependencies and Blockers

- `uv lock --check` and `uv pip check` passed; all runtime and verification-tool imports succeeded. The offline integration fixture verified CRFsuite fitting/marginals and joblib prediction/confidence equivalence after loading. No cross-platform bitwise reproducibility is claimed.
- Editable installation imports work from outside the repository with isolated Python (`-I`). Wheel inspection verifies `crf` and `py.typed` without legacy/data packages. Source distributions explicitly include only code, tests, and documentation, excluding raw corpus files.
- Ignore-rule checks passed for interim/processed datasets, model bundles, reports, distribution builds, and weights. Raw gold/literature inputs, intentional fixtures, and `uv.lock` remain visible. Existing raw files and generated datasets/models/reports were not overwritten.
- Unrelated pre-existing failures: `scripts/fetch_dictionary.py` fails Ruff import ordering and formatting, with the same findings in baseline and final checks. It remains unchanged; core type/tests and all changed-code Ruff checks pass. No checks were weakened.
- The pre-existing `AD main.py` index/worktree state was left untouched.
- Production corpus availability and tokenizer compatibility are unverified; the corpus gate cannot be completed until they are established.
- The production raw-text adapter is still incomplete: upstream standalone format-character omissions need handling, UD token boundaries require adaptation, and the shared aligned normalizer needs integration with bounded paragraph streaming. Corpus readiness, full model training, and final evaluation remain unperformed.
