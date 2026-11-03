# Implementation History

Dated records of what was built, requirement reviews, verification and limitations. This is evidence of past work, not a design specification or a guarantee of current behavior. See [ARCH.md](ARCH.md) for design/contracts, [PLAN.md](../PLAN.md) for goals/status, and [AGENTS.md](../AGENTS.md) for operating procedures.

## Milestone 1

**2026-10-28 — Environment and core package.** Established installable `src/crf/`, the uv-managed CPython 3.14.6 environment on macOS arm64, packaging, artifact exclusions and low-level feature/tokenizer foundations. The locked environment included `tokenize-uk==2.0.0` without spaCy, `python-crfsuite==0.9.12` and `sklearn-crfsuite==0.5.0`; reproduce the recorded environment from its matching `uv.lock`.

Retired incompatible legacy corpus-frequency/±1 features, SVD/clustering and whole-file spaCy tokenization. The new core uses direct CRFsuite dictionaries; old imports, keys, CLI switches and exploratory model artifacts are unsupported. The text-only tokenizer preview preserves original upstream spans and creates no training artifacts.

Verified 2026-10-28: 20 tests and strict types (7 core files) pass; environment imports, packaging and tiny CRF persistence were checked. Review covered migration compatibility, surface-only features and tokenizer spans. Repository lint/format retain the unrelated failure recorded in the verification section below; no production experiment was run.

### PyCharm Environment

**2026-10-28.** Aligned the native uv SDK, project/module roots and shared preview/test configurations. The IDE pyproject module model had regenerated roots relative to `.idea`, so it was disabled; active setup instructions remain in [PYCHARM.md](PYCHARM.md).

Verified XML/paths, native uv SDK/imports and preview CLI; disabled the IDE module model that rewrote roots. SDK backup preserved; restart required to reload changes. UI Run/Debug/indexing remain unverified (connection timeouts). See [PyCharm setup](PYCHARM.md).

## Milestone 2

**2026-10-28 — Configuration, records and lifecycle.** Built strict configuration/path validation, immutable sentence/input records, streaming JSON/identity helpers, no-clobber run allocation/publication, stderr logging and aggregate RSS monitoring.

Completed 2026-10-28 with temporary offline fixtures; changed-code lint/format passed, with the repository-wide exception below. Reviews/tests covered strict configuration/path boundaries, schema/identity/hash integrity, no-clobber lifecycle races and sticky RSS failures. Stage-specific validators and pipeline integration remain later work.

| Task | New tests | Full suite | Typed core files |
|---|---:|---:|---:|
| Configuration/paths | 88 | 108 | 9 |
| Records/serialization/identity | 119 | 227 | 12 |
| Lifecycle/logging/resources | 47 | 274 | 15 |

## Milestone 3

**2026-10-28–29 — Gold loading, label space, normalization and features.** Built the streaming CoNLL-U parser and separate label-free audit, shared service-tag mapping, aligned versioned normalization/gold-record conversion, and strict versioned sentence-local feature extraction/validation. Preserved original forms/labels and explicit final-test authorization. Existing default extractor calls and valid feature dictionaries remain compatible; incorrect normalization/schema inputs now fail explicitly.

Completed with offline fixtures; changed-code lint/format passed at every step, with the repository-wide exception below. Pre-test reviews covered parser/audit isolation, source/target inventory consistency, normalization/alignment and exact feature/context schemas. Compatibility boundaries and remaining integration work are specified in [ARCH.md](ARCH.md).

| Task | Date | New tests | Full suite | Typed core files |
|---|---|---:|---:|---:|
| Gold parser/audit | 2026-10-28 | 59 | 333 | 16 |
| Shared tag mapping | 2026-10-28 | 89 | 422 | 17 |
| Aligned normalization/record conversion | 2026-10-28 | 73 | 495 | 18 |
| Feature schema/context | 2026-10-29 | 99 | 594 | 19 |

M3 is complete for fixture-based acceptance; real-corpus readiness, the production raw-text/UD adapter and model/orchestration stages remain pending. Original inputs/artifacts and unrelated `main.py` state were preserved; no real-corpus training/evaluation is claimed.

## M4 Task 1 — Literature Reader and UD Adapter

**2026-10-30 — Complete for fixture-based acceptance.** Added shared flat guarded discovery and literature-only identities in `crf.data.identity`, strict incremental decoding/bounded paragraphs and sentence/exclusion events in `crf.data.literature`, and aligned versioned UD adaptation in `crf.data.segmentation`. Whole-sentence batches enforce token bounds. Existing low-level tokenizer calls and record schemas remain compatible. No raw inputs or generated artifacts are written.

Pre-test specification review:

- Discovery → immediate exact `.txt` regular files in filename order; shared path checks reject symlinks, escapes and physical gold aliases without opening gold. Empty/missing sources fail; directories and other suffixes are ignored. Text-content contamination remains task 2.
- Decoding/streaming → strict configured incremental codec, bounded byte reads, CRLF preserved across chunks, blank-line paragraphs, EOF flush and no cross-file joins. Lines/paragraphs exceeding limits are counted and drained without retaining their text; delimiter whitespace is excluded from paragraph volume.
- Alignment/UD → normalization before abbreviation-aware sentence segmentation, original absolute code-point spans and forms, case-preserving feature inputs, punctuation/hyphen/abbreviation-period separation, internal apostrophes and grouped digits retained. Backend omissions are recovered from source slices. Adapter/package/normalizer/Unicode versions are exposed as immutable metadata; run persistence belongs to orchestration.
- Resource policy → paragraph and sentence limits use original spans, tokens are counted after adaptation, rejected candidates remain whole, diagnostics are streamed, batching preserves order and complete sentences. Gold records are never filtered or parsed here.
- Integrity/failures → identity verified before/after decoding and consumed-byte hash checked at EOF; prior yields are provisional until exhaustion. Check invalid/truncated encoding, changed sources, unsupported versions, empty/BOM/formatting-only inputs, exact limits and one-over, buffer sizes down to one byte, CR/LF/CRLF and combining marks at boundaries, oversized recovery and bounded reads/retention.

Verification on CPython 3.14.6/macOS arm64, using the existing locked environment. Every command below used `UV_CACHE_DIR=/private/tmp/postag-uv-cache` because the default cache was sandbox-inaccessible:

| Check | Actual command (after the cache environment prefix) | Result |
|---|---|---|
| Focused | `uv run --frozen pytest tests/test_literature.py tests/test_segmentation.py -q` | Passed: 86 new cases |
| Lint | `uv run --frozen ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001, reproduced before and after implementation |
| Formatting | `uv run --frozen ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py`; all 43 other Python files passed |
| Types | `uv run --frozen mypy src/crf/` | Passed: 21 source files |
| Tests | `uv run --frozen pytest tests/ -v` | Passed outside the sandbox: 680 tests |
| Changed-code lint/format | `uv run --frozen ruff check src/crf/data/identity.py src/crf/data/segmentation.py src/crf/data/literature.py tests/test_literature.py tests/test_segmentation.py` and the same paths with `ruff format --check` | Passed: all five changed Python files |

The initial sandboxed full suite had 679 passes and one failure in unchanged `test_real_process_monitor_smoke`: macOS denied psutil's `sysctl` process enumeration. The isolated smoke test passed outside the sandbox, followed by the complete 680-test suite there. No monitor code or checks were weakened. The pre-existing `AD main.py` state and failing dictionary script were untouched; scoped diffs confirm no edits to that script or the monitor/test.

Tests cover one-byte reads, three newline styles, EOF, BOM/normalization alignment, three alternative source codecs, invalid/truncated bytes, whole-candidate limits/recovery, path/gold isolation, source changes, formatting omissions, stressed apostrophes, digit grouping, bounded read sizes/retention and ordered batches. Pre-test review preceded new tests; an initial mutation fixture was changed to equal-length valid text to isolate checksum failure from a legitimate earlier decoding error. Bounded-read instrumentation was restricted to the input file so backend metadata reads remained independent.

Limitations: tokenizer compatibility with the official selected gold release remains a corpus-readiness gate. No production corpus was processed, no training/evaluation ran, and no run metadata was published. Contamination/deduplication/known-source/MWT policy and aggregate-monitor integration remain M4 tasks 2–3; callers must exhaust preprocessing iterators and collect their exclusion events before publishing successful outputs.

## M4 Task 2 — Contamination Controls

**2026-10-30 — Complete for fixture-based acceptance.** Added `crf.data.contamination`: a text-only contamination layer with immutable decisions/reports, single-use mutable SQLite audit state, no-clobber intermediate allocation and automatic source/configuration/preprocessing identities. No dependency, configuration-schema or existing reader/record API changes were required. A committed `e.compile` typo prevented adapter imports; reproduced with the existing segmentation test module and repaired to `re.compile`. The existing 86 reader/adapter tests passed after that repair.

Pre-test specification review:

- Fingerprints → shared versioned normalization followed by Unicode whitespace collapse/trim and exact SHA-256 of UTF-8; preserve case, punctuation, stress/formatting characters and embedded BOMs. Reject empty/unsupported inputs. Use supplied gold text or the audit reader's surface reconstruction, never a syntactic-word join.
- Gold isolation → exhaust `iter_gold_audit()` for all three splits before filtering; only text/identifiers/spacing are exposed. Store exact-overlap fingerprints per split; harvest MWT boundary patterns only from train range rows. No labeled loader, target mapping, predictions or metrics are used. Gold raw limits never apply.
- Streaming/order → disk-backed gold/MWT/seen indexes with bounded SQLite cache; process filename/sentence order from captured input identities. Retain the first otherwise eligible occurrence. Whole-source evidence rules precede raw limits, exact overlaps, train MWT patterns and duplicate decisions. Excluded candidates do not reserve duplicate slots.
- Source/MWT policy → validate explicit literature paths, held-out split references and evidence; freeze canonical rules in metadata. Known-source files are excluded in full without decoding. Match train MWT normalized token sequences across punctuation spacing, preserving case and word boundaries; do not expand surfaces or harvest held-out MWT patterns. Near duplicates and unresolved provenance remain disclosed.
- Lifecycle/diagnostics → reserve a fresh index under `data/interim/<run_id>/`, already covered by `.gitignore`; preserve existing artifacts and all raw inputs. Incrementally emit reason/split/MWT/duplicate/evidence diagnostics and original resource events. Reports include counts, text/byte volumes, source checksums, versions/configuration hash and deterministic accepted-corpus identity only after final source-set verification. SQLite errors, malformed gold, changed/added inputs, early termination and reuse/closed-state access must not authorize success.

Verification on CPython 3.14.6/macOS arm64. Every command used the task-local `UV_CACHE_DIR=/private/tmp/postag-uv-cache` prefix with the existing locked environment:

| Check | Actual command (after cache prefix) | Result |
|---|---|---|
| Focused | `uv run --frozen pytest tests/test_contamination.py tests/test_literature.py tests/test_segmentation.py -q` | Passed: 144 cases, including 58 new contamination cases |
| Lint | `uv run --frozen ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001; reproduced before/after implementation |
| Formatting | `uv run --frozen ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py`; other 45 Python files passed |
| Types | `uv run --frozen mypy src/crf/` | Passed: 22 source files |
| Tests | `uv run --frozen pytest tests/ -v` | Passed: 738 tests outside the sandbox, retaining access needed by the existing RSS smoke test |
| Changed-code lint/format | `uv run --frozen ruff check src/crf/data/contamination.py src/crf/data/segmentation.py tests/test_contamination.py` and the same paths with `ruff format --check` | Passed: all three changed Python files |

Tests verified exact normalization/hash expectations, case/punctuation/formatting distinctions, all-split overlap, supplied-text precedence and MWT reconstruction/empty-node omission, invalid annotations left uninterpreted, train-only boundary-aware MWT matching, known-source skipping without decoding/dedup slot consumption, deterministic reports and first occurrences, rule/evidence/config/input identities, exclusion volume, fresh indexes, protected existing outputs/symlinks, raw-limit passthrough, single-use/early-close lifecycle, malformed gold, decoding/SQLite errors and changed/added/removed inputs. Initial fixtures that assumed a lowercase word after a period would force a sentence boundary were corrected to use explicit paragraph delimiters after inspecting the existing backend output; segmentation rules were not tuned to the fixtures.

Scoped diffs confirm the dictionary script, memory monitor/tests and pre-existing `AD main.py` state were untouched. New indexes were produced only in temporary offline test repositories; `.gitignore` already covers their prescribed `data/interim/` location. No production corpus, model inference/training or final evaluation has run. Raw standalone inference remains unfiltered. Run-report publication and aggregate-monitor integration remain with later orchestration/M4 resource work. Official source provenance and residual near-duplicate/tokenizer limitations remain corpus-readiness gates.

## M4 Task 3 — Streaming and Resource-Failure Verification

**2026-10-30 — Complete for fixture-based acceptance.** Prior contamination work was committed as `9ba0b64`, with the unrelated staged `main.py` entry preserved. Added compatible keyword-only optional monitor parameters through source identities/hashing, gold/raw readers, batching and contamination auditing. The caller owns the monitor lifecycle; configuration-aware calls require its budget to match. Added shared active-monitor checkpoints and retained unmonitored bounded-library behavior. Existing focused tests passed (203 cases) and strict types passed before adding new verification. No dependency, configuration or record/report schema changes were required.

Pre-test requirement review:

- Streaming → checkpoints on bounded opaque-byte hashing and raw decoding, including oversized-paragraph draining; gold physical-line and complete-sentence boundaries; raw adaptation and token-bounded batches. No whole-corpus text/features/marginals are accumulated. SQLite indexes grow incrementally on disk. A test-only streamed JSONL intermediate will exercise existing atomic publication without introducing a pseudo-label schema before M6.
- Candidate integrity → character/token limits reject whole raw candidates, recover at genuine paragraph/sentence boundaries and preserve source spans/ordinal gaps. Gold parsing/audit never uses raw limits or truncation. A resource failure aborts rather than filtering a gold sentence.
- Policy boundaries → raw inference retains repeated, gold-overlapping and known-MWT sentences subject only to resource limits; pseudo-label preprocessing applies the separate contamination policy. Neither path applies model-confidence selection here; confidence and sampling remain M6 work.
- Controlled failures → deterministic sampler inputs check budget equality/exceedance, sampling failure, source-hashing/raw-draining/gold-read/audit checkpoints and latched failure. Inactive/finished/wrong-budget monitors must fail before outputs. Check failures after partial output, during final monitor exit and after audit exhaustion: no successful report/publication may bypass the shared monitor's failure. Production RSS safety remains best-effort; tests do not induce real low-memory conditions.
- Lifecycle → optional parameters preserve existing calls and schemas; readers never start/stop a caller's monitor. Integration fixtures must exhaust iterators, close DB/streams, exit monitoring, require `ensure_success()` and publish only afterward. Early termination or exceptions leave no published JSONL/completion result; existing artifacts/raw sources remain intact. Aggregate RSS behavior uses the existing separately tested sampler and controlled values in integration tests.

Verification on CPython 3.14.6/macOS arm64. Commands used `UV_CACHE_DIR=/private/tmp/postag-uv-cache` with the locked environment:

| Check | Actual command (after cache prefix) | Result |
|---|---|---|
| Focused integration | `uv run --frozen pytest tests/test_preprocessing_streaming.py -q` | Passed: 23 new offline integration cases |
| Lint | `uv run --frozen ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001 only |
| Formatting | `uv run --frozen ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py`; other 46 Python files passed |
| Types | `uv run --frozen mypy src/crf/` | Passed: 22 source files |
| Tests | `uv run --frozen pytest tests/ -v` | Passed: 761 tests outside the sandbox, including the existing real RSS smoke test |
| Changed-code lint/format | `uv run --frozen ruff check src/crf/utils/resources.py src/crf/data/identity.py src/crf/data/conllu.py src/crf/data/literature.py src/crf/data/contamination.py tests/test_preprocessing_streaming.py` and the same paths with `ruff format --check` | Passed: six changed Python files |

The new tests verify exact-budget success, latched budget/sampler failures, bounded hashing, aborting an oversized-paragraph drain before emitting a fragment, fatal mid-gold-sentence failures, full 24-word gold records despite four-token raw limits, complete raw rejection/recovery with original spans, inference versus pseudo-label exclusions, token-bounded batching with one-candidate lookahead, inactive/finished/wrong-budget monitor rejection, report checks against final monitor outcomes, partial SQLite construction cleanup, early termination and late decoding failure cleanup. A 300-sentence stream writes diagnostics and grows the disk index incrementally while weak-reference observation verifies at most four live raw sentence objects; prior records are not retained. Atomic JSONL publication is blocked by failures after partial writes or during final monitor exit. Controlled samplers avoid real memory exhaustion; original fixture sources remain unchanged during processing.

The dictionary script's lint/format failures are unchanged from the previous task; no checks were weakened or unrelated files repaired. `.gitignore` already covers test intermediates in the prescribed artifact tree. No production corpus, training or final evaluation ran. M4 is complete at fixture level; selected-corpus tokenizer/source compatibility, M5–M8 model/CLI stages and M9 full pipeline integration remain pending. Memory monitoring remains best-effort during indivisible library/native operations; future stage owners must supply the monitor, exhaust all iterators and require successful monitor exit before guarded publication.

## Refactoring and Refinement

**2026-10-30 — Team analysis and implementation plan.** Three reviewers inspected preprocessing/contamination, configuration/artifact/resource foundations, and CLI/features against ARCH and PLAN. Priorities are a silently failing monitor wait, repeated same-prefix MWT candidate work, a long configuration parser, and an unvalidated legacy dictionary download destination. Preserve the staged/deleted `main.py` and pre-existing untracked `tests/test_preprocessing_streaming.py`.

Selected increments: fail-closed monitor scheduling plus section-parser extraction; indexed MWT probes plus explicit exclusion branches; typed reusable dictionary retrieval with a thin CLI and validated metadata/no-clobber publication. Avoid rewriting the bounded character scanner, gold parser or normalization engine without profiling evidence. Retain independent feature validation and repeated source-integrity scans; these enforce contracts. M5–M9 and production experiments remain outside this refactoring scope.

Baseline checks using `UV_CACHE_DIR=/private/tmp/postag-uv-cache`: `uv run ruff check .` failed only on dictionary-script I001; `uv run ruff format --check .` failed only on that script (46 other files passed); `uv run pytest tests/ -q` produced 760 passes and one failure because sandboxed macOS process enumeration denied `sysctl` in the existing real-RSS smoke test. This is an environment restriction, not evidence of a monitor implementation regression.

Implementation requirement reviews and final verification follow below as increments complete.

Pre-test requirement review — foundations and contamination:

- Monitor: validate positive finite platform-supported sampling intervals, including huge integers without overflow. Guard both waiting and sampling; latch scheduling errors with their cause, retain earlier failures and interrupt the owner. Exit/publication must reject latched failures. Test the platform boundary, oversized/huge values and a controlled scheduling exception without real memory exhaustion.
- Configuration: extract typed data/path/codec and training parsers while retaining defaults, schema, serialization and path guards. Some multiply-invalid configurations may report a different first invalid field; error category and valid configuration behavior remain compatible. Existing focused configuration/resource checks passed (109 cases, real-RSS smoke excluded there).
- Contamination: extract train-only MWT indexing; retain first observed surface, exact normalization and insert-or-ignore behavior. Probe distinct eligible lengths in descending order and exact encoded keys, retaining earliest token position and the previous matching-prefix choice (JSON comma sorts before closing bracket). Check monitoring between probes. Explicit gold-overlap, MWT and duplicate branches preserve priority and exclusion fields. Existing 58 contamination tests passed before new tests. New cases will cover competing lengths/positions, same normalized surfaces, same-head distractor work bounds and fatal monitor checkpoints.

The baseline real-RSS smoke test was rerun outside the sandbox with `UV_CACHE_DIR=/private/tmp/postag-uv-cache uv run pytest tests/test_resources.py::test_real_process_monitor_smoke -q`: passed (1 test), confirming the baseline failure was sandbox process-enumeration access.

**2026-11-03 — Resumed implementation and pre-test dictionary review.** Existing foundations and contamination changes were retained and reviewed; the user-owned streaming test still has SHA-256 `833a300477e4695a2eb3f7dd0d7f9520cf1225d4409b8d66c45a211c255b3ab4`. Added monitor boundary/scheduling regressions (113 focused passes excluding real RSS) and contamination matching regressions (86 contamination/streaming integration passes). Independent review found no precedence or match-selection regression. The operation-count fixture verifies one Python encoding for 80 same-head/same-length distractors; SQLite may still scan index entries, so this is not an end-to-end speedup claim.

Dictionary requirement mapping before new tests: frozen metadata validates required strings, basename, SHA-256 and credential-free HTTPS; strict JSON rejects malformed/duplicate/nonfinite inputs while allowing extra metadata fields. Manifest/output locations remain confined to `data/dict_uk`; directory-anchored no-follow descriptors reject symlink redirection, and regular-file checks reject special files. Stream download bytes to a fresh temporary file, verify the checksum, hard-link without replacement and clean temporary entries on failure. Redirects must retain HTTPS. The argparse CLI delegates to the core and reports missing manifests on stderr with failure status. Planned offline tests cover malformed paths/URLs/metadata, redirects, existing matches/mismatches/symlinks/FIFOs, streaming/checksum/read/publication failures, winner preservation and CLI behavior. No live download is authorized by these tests. `.gitignore` now excludes dictionary downloads and temporary entries while allowing the default source manifest; `git check-ignore` confirmed both cases.

Independent dictionary review identified that `http.client.IncompleteRead` is an `HTTPException` outside the CLI's `OSError`/`ValueError` handler. Temporary cleanup already covers it; the targeted repair extends CLI handling to HTTP protocol errors. Regression expectations remain controlled stderr/exit failure and no partial publication after a truncated response.

**2026-11-03 — Completed and verified.** The HTTP error was reproduced offline, repaired, and covered by CLI/stream-cleanup regressions. The dictionary suite passes 48 offline cases. Foundations added four cases and contamination added five, for 57 additional cases overall. Valid pipeline configuration/data schemas remain unchanged. Dictionary CLI stdout now contains only the result path; status diagnostics go to stderr. Unsafe manifests/URLs and unsupported monitor intervals now fail explicitly. Multiple-invalid-field configuration errors may be reported in a different order.

Final verification on CPython 3.14.6/macOS with `UV_CACHE_DIR=/private/tmp/postag-uv-cache` prefixed to each command:

| Check | Actual command | Result on 2026-11-03 |
|---|---|---|
| Lint | `uv run ruff check .` | Passed; prior dictionary I001 exception resolved |
| Formatting | `uv run ruff format --check .` | Passed; all 49 Python files formatted |
| Types | `uv run mypy src/crf/` | Passed; 23 source files |
| Tests | `uv run pytest tests/ -v` | Passed; 818 tests in 5.17 seconds, outside sandbox for real-RSS process enumeration |

Focused checks also passed: configuration/resources (113 cases with the real-RSS smoke test deselected); contamination/preprocessing streaming (86); `uv run pytest tests/test_dictionary.py -q` (48). Scoped lint/format checks passed during each increment. Local documentation links and `git diff --check` passed; final index inspection and the streaming-test checksum confirm user changes were preserved. No dependencies, raw datasets or existing generated artifacts changed; no Git mutations were performed.

Remaining limits: default dictionary manifest is absent, so no real release/download was verified; HTTP behavior was tested with offline responses. MWT improvement is verified by reduced Python encoding work, not production benchmarks. Memory monitoring remains best effort. Official-corpus compatibility, M5–M9 implementation and real training/evaluation remain separate pending work. Defer scanner/parser/normalizer rewrites and feature extraction optimization until profiling justifies them; this refactoring does not weaken their validation or source-integrity checks.

## M5 Task 1 — Single-Candidate CRF Training

**2026-11-03 — Implementation and pre-test requirement review.** Added `crf.models.training.fit_candidate` and the training interface contract in ARCH. Work is limited to the first M5 task; serialization, metrics and search remain pending.

- Input isolation → consume only the supplied ordered validated records; reject validation/test roles, gold path/mapping mismatches, unaligned/empty sentences and unrequested pseudo records. Require supervised data even in augmented mode. Feature extraction receives forms alone; native fitting receives no development arguments. Hash canonical ordered JSONL bytes and record total/pseudo counts. Producers remain responsible for full source/span validation and later publishing exactly these dataset bytes.
- Optimization → fresh L-BFGS estimator per call, configured regularization/grid and prescribed defaults, explicit line-search defaults, no fabricated random state. Main-thread execution and a process-local lock prevent overlapping fits. Capture complete estimator options, configuration identity, seed and environment versions. Native training data stays in memory; it is never silently reduced.
- Termination → inspect raw loss messages because the dependency parser does not recognize nonfinite loss text. Require consecutive iterations with finite objectives and recognized stop evidence. Maximum-iteration stops are budget-limited; errors/unknown stops fail even if native code writes a model. A local tiny-data probe confirmed that the pinned library writes a file after error code (2), so this case must be rejected. Zero configured iterations preserves the dependency's unlimited convention.
- Resources/lifecycle → checkpoints while preparing features, loading native examples, receiving optimizer messages and after fitting, plus aggregate background monitoring. Own a configured monitor or accept a matching active enclosing monitor. Shared-monitor results remain provisional until successful owner exit. Failures clean only the fit's temporary native file and close its tagger; no model/report is published. Successful estimators retain no monitor/callback closure. Native interruptions remain best effort, not a hard RSS bound.
- Planned verification → tiny real two-label fitting, exact settings/checksums/counts, fresh fits, custom mapping and explicit augmentation; deterministic optimizer logs for convergence/limits/invalid losses/errors; held-out/empty/late input failure and inactive/wrong-budget/resource failures. No production data, network or persistent training outputs are needed.

**2026-11-03 — Complete for fixture-based acceptance.** Added 47 offline cases, including native integration fixtures using the locked CRFsuite dependencies. Verified exact estimator options, declared records/features/targets with no held-out arguments, order-sensitive dataset checksums, fresh models, mapped training, explicit augmentation, convergence/iteration limits, optimizer errors despite native file creation, malformed/nonfinite log evidence, main-thread/sequential guards, late iterator errors, and inactive/finished/wrong-budget monitors. Controlled RSS failures during preparation, native append, optimizer callbacks, after fit and owned-monitor exit all prevent a successful result; failed native files are removed. The successful owned-monitor path completes monitoring before returning. Shared-monitor callers still own final publication checks.

Verification on CPython 3.14.6/macOS, with `UV_CACHE_DIR=/private/tmp/postag-uv-cache` prefixed to every command:

| Check | Actual command | Result on 2026-11-03 |
|---|---|---|
| Focused | `uv run pytest tests/test_training.py -q` | Passed: 47 cases |
| Lint | `uv run ruff check .` | Passed |
| Formatting | `uv run ruff format --check .` | Passed: 51 Python files |
| Types | `uv run mypy src/crf/` | Passed: 24 source files |
| Tests (sandbox) | `uv run pytest tests/ -v` | 864 passed; existing real-RSS smoke failed because macOS `sysctl` process enumeration was denied |
| Tests (process access permitted) | `uv run pytest tests/ -v` | Passed: all 865 tests in 4.12 seconds |

`git diff --check` and local documentation references passed inspection. Generated model/dataset paths remain ignored; no production datasets, persistent model bundles or dependency changes were produced. The pre-existing `AD main.py` index/worktree state is unchanged; no Git mutations were performed. Memory limits remain best effort and native optimization/environment changes may affect results. Source/span validity is delegated to existing record factories; serialization, strict bundle loading, metrics/search and real-corpus training remain separate pending tasks.

## M5 Task 2 — Model Bundle Persistence

**2026-11-03 — Implementation and pre-test requirement review.** Added strict bundle metadata and trusted-local joblib persistence, with an embedded immutable training record in freshly fitted estimators. Promoted parameter/environment helpers for use by training and persistence. Added lexical-only configuration parsing for recorded original paths; ordinary configuration resolution remains the default.

- Sidecars → exact top-level fields, complete canonical configuration, frozen-grid/full estimator settings, sorted fixed inventory, seed, positive/aligned supervised/pseudo counts, checksums and valid termination/iteration limits. Require exact installed Python/Unicode/runtime dependency versions. Embedded training evidence and model options/log/learned labels must agree with sidecar; learned labels may be a subset of the fixed inventory.
- Save lifecycle → require successful finished training monitoring; reserve a fresh prescribed candidate directory without following symlinks. Serialize under a separate configured monitor, compare deterministic synthetic predictions and every full-precision marginal after loading, dispose the restored estimator, and exit monitoring successfully before publishing the model. Publish metadata last. Partial errors may leave an incomplete directory/model, but cannot authorize loading; existing outputs are preserved.
- Loading → read regular no-follow files through anchored directory descriptors; parse metadata/environment and optional frozen references before deserialization. Copy model bytes incrementally to an anonymous temporary snapshot, verify checksum, then deserialize that same snapshot. Validate estimator/evidence/log and close rejected temporary native state. Standalone copies require no original repository or dataset; repository-bound loads additionally verify the prescribed containing path. Joblib remains trusted-local pickle, not authenticated/safe untrusted deserialization.
- Boundaries/failures → planned offline tests cover unmapped/mapped/iteration-limited round trips, Viterbi confidences, portable loading with absent/inaccessible sources, incomplete/corrupt/swapped bundles, schema/type/environment/parameter/evidence mismatches, symlinks/special files, no-overwrite and controlled serialization/resource failures. New serialization memory is monitored best effort; callers of a shared loading monitor must still require its successful exit. No production artifacts or data are needed.

**2026-11-03 — Complete for fixture-based acceptance.** Added 98 offline persistence/configuration cases. Native round trips preserve predictions, every unrounded marginal and Viterbi-label confidences for default/mapped label spaces and iteration-limited/unlimited settings. A copied bundle loads and predicts after the source fixture's entire data/model tree is removed. Guarded metadata parsing verifies no source-path resolution/stat/open calls; installed dependency metadata remains intentionally readable for compatibility checks. Verified checksum-before-deserialization, exact frozen references, embedded evidence, strict complete JSON/configuration, unknown/duplicate fields, invalid Unicode/types/numbers/versions/paths, missing/corrupt models, wrong serialized object types and special-file/symlink rejection.

Failure fixtures verify dump errors, changed round-trip results, serialization RSS failure, serialization/load monitor exit failure, failed sidecar publication, and cleanup of rejected native files. Fresh-directory allocation preserves existing complete and incomplete bundles and rejects redirected model directories. The verified model snapshot remains stable if its source changes after checksum verification. Shared loading monitors remain caller-owned/provisional; callers must close loaded bundles when done. Existing training/configuration regressions passed (229 focused cases before the final four boundary tests were added).

Verification on CPython 3.14.6/macOS, with `UV_CACHE_DIR=/private/tmp/postag-uv-cache` prefixed to each command:

| Check | Actual command | Result on 2026-11-03 |
|---|---|---|
| Focused | `uv run pytest tests/test_bundles.py -q` | Passed: 98 cases |
| Lint | `uv run ruff check .` | Passed |
| Formatting | `uv run ruff format --check .` | Passed: 53 Python files |
| Types | `uv run mypy src/crf/` | Passed: 25 source files |
| Tests | `uv run pytest tests/ -v` | Passed: all 963 tests in 7.94 seconds, with process-enumeration access for the existing RSS smoke test |

`git diff --check` passed; new documentation references and the bundle artifact exclusions were inspected. Dependencies, original datasets and existing generated outputs are unchanged. All bundles were created in temporary test repositories. No Git mutations were performed; the original `AD main.py` state and concurrent PyCharm XML/guide/history changes were preserved. Bundle loading intentionally requires the new embedded training evidence; exploratory joblib files are not production bundles. Exact environment matching is conservative, joblib requires trusted local artifacts, and memory monitoring is best effort. Validation metrics, candidate search and real-corpus runs remain pending.

## M5 Task 3 — Validation Metrics

**2026-11-03 — Pre-test requirement review.** Implemented validation-only aligned scoring, a shared immutable gold-training lexical vocabulary, fixed-inventory confusion/F1 calculation and strict report validation/publication. Pooled token counts include punctuation/SYM/X; absent tags receive zero F1. OOV uses normalized lexical forms from supervised training only; non-punctuation masks use original gold UPOS. Empty diagnostic subsets retain zero counts and null accuracy. Sentence identity/order, token IDs/forms/lengths, mappings and held-out roles are checked before accepting results. Report validation recomputes primary values from matrix counts, checks diagnostic subset bounds and permits only serialization-level floating-point differences. Publication requires finished successful matching-budget monitoring and a fresh prescribed validation path. Record producers retain source/schema validation responsibility; actual bundle and trial-link verification remains orchestration work. Planned fixtures cover manual counts/F1, mapped inventories, masks, empty subsets, alignment failures, strict report rejection and publication guards.

**2026-11-03 — Completed for offline acceptance.** Added 34 deterministic cases covering hand-calculated pooled accuracy, fixed 17/14-class macro-F1, per-tag counts and matrix orientation, shared case-normalized gold-only OOV masks, original-UPOS punctuation exclusions, empty diagnostic subsets, SYM/X inclusion, mapping/alignment and held-out-role failures, inconsistent aggregate reports, vocabulary reuse/configuration binding, tolerant float comparison and no-clobber/unfinished-monitor publication. Test development exposed a missing test import; it was corrected before final verification. No real inference, corpus evaluation, model selection or dependency changes occurred. Final-test evaluation and bundle/trial reference verification remain orchestration work; aggregate reports cannot reconstruct diagnostic membership independently of aligned scoring.

Verification on CPython 3.14.6/macOS on 2026-11-03, each command prefixed with `UV_CACHE_DIR=/private/tmp/postag-uv-cache`: `uv run pytest tests/test_evaluation.py -q` passed 34 cases; `uv run ruff check .` passed; `uv run ruff format --check .` passed (55 Python files); `uv run mypy src/crf/` passed (26 source files); `uv run pytest tests/ -v` passed 997 tests with process-enumeration access for the RSS smoke test. Documentation links/artifact exclusions were inspected and `git diff --check` passed. Original data and existing artifacts were preserved; reports were generated only in temporary test repositories. Existing `AGENTS.md`/Git-policy history edits and staged `main.py` were left intact; no Git mutations were performed.

## Verification Record

M1–M3 code checks: 2026-10-29, CPython 3.14.6/macOS arm64. These historical results were not rerun for the documentation compaction on 2026-10-30; references, task state and scoped diffs were checked. Latest code verification is recorded in [M4 task 3](#m4-task-3--streaming-and-resource-failure-verification), with earlier M4 results above.

| Check | Command | Recorded result |
|---|---|---|
| Lint | `uv run ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001; changed-code lint passed |
| Formatting | `uv run ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py` requires formatting; changed-code formatting passed |
| Types | `uv run mypy src/crf/` | Passed (2026-10-29): 19 source files |
| Tests | `uv run pytest tests/ -v` | Passed (2026-10-29): 594 tests, including 110 feature tests, 73 normalization/record cases, 89 shared label-space tests, 59 gold parser/audit tests and two offline integration tests |

## Recorded Limitations

- Runtime imports, `uv lock --check`/`uv pip check`, isolated editable imports, wheel/sdist contents and artifact ignore rules were verified during M1–M2. Tiny offline CRF fitting/marginal/joblib checks passed; cross-platform bitwise reproducibility is not claimed.
- Historical lint/format exception: `scripts/fetch_dictionary.py` failed I001 and formatting during M1–M4. Resolved by the [2026-11-03 refactoring](#refactoring-and-refinement); earlier results remain preserved above. Existing `AD main.py` state remains untouched.
- Official gold availability/release identity, source separation and selected-corpus tokenizer compatibility remain unverified. M4 covers bounded streaming, aligned UD adaptation/formatting-character preservation, contamination controls and resource-failure integration with offline fixtures. No production training or final evaluation has run.

## Documentation Maintenance

- **2026-11-03 — Git procedure update:** allowed task-scoped staging and commits without additional confirmation and removed the requirement to supply suggested commit messages/commands. Retained authorization for other Git mutations, unrelated-change preservation and artifact exclusions; aligned the artifact pre-commit check wording. Verified the documentation diff and consistency by inspection; no executable behavior changed or Python checks were required.

- **2026-10-30 — Report compaction:** shortened milestone narratives/tables in ARCH/PLAN; verified all 34 checklist entries, pending milestone/corpus sections, architecture tables/code fences and local links. No code checks were rerun.
- **2026-10-30 — Documentation separation:** moved build/verification records here; kept architecture design in ARCH, goals/status in PLAN and placement rules in AGENTS. Verified migrated evidence, goal statuses, design tables/code examples, local references and scoped diffs. No code, dependency, dataset or generated-artifact changes; no code checks rerun. The pre-existing `AD main.py` state was preserved.

## PyCharm Project Root Correction

**2026-11-03.** Made the module content root explicitly `$PROJECT_DIR$` (the `postag` checkout), retained only repository `src`/`tests` roots and indexing exclusions, removed sibling VCS mappings and inherited the existing project SDK instead of the module override `uv (git)`. The previous module-relative content URL already resolved to `postag` on disk; this correction removes ambiguity and stale settings rather than relocating files. Updated the setup guide.

Requirement review and verification: parsed changed IDE XML, resolved the module/content/source paths to this checkout, and checked exclusions and the single project Git mapping. No application code or runtime configuration changed; the four Python checks were not rerun for IDE-only settings. Live PyCharm inspection timed out; reopen `~/git/postag` to reload settings, with live UI confirmation still pending. Existing unrelated working changes were preserved.
