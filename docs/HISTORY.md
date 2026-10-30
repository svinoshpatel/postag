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
- Lint/format exception: unchanged `scripts/fetch_dictionary.py` fails I001 and formatting, reproduced before/after implementation. Changed-code checks pass; no checks were weakened. Existing `AD main.py` state remains untouched.
- Official gold availability/release identity, source separation and selected-corpus tokenizer compatibility remain unverified. M4 covers bounded streaming, aligned UD adaptation/formatting-character preservation, contamination controls and resource-failure integration with offline fixtures. No production training or final evaluation has run.

## Documentation Maintenance

- **2026-10-30 — Report compaction:** shortened milestone narratives/tables in ARCH/PLAN; verified all 34 checklist entries, pending milestone/corpus sections, architecture tables/code fences and local links. No code checks were rerun.
- **2026-10-30 — Documentation separation:** moved build/verification records here; kept architecture design in ARCH, goals/status in PLAN and placement rules in AGENTS. Verified migrated evidence, goal statuses, design tables/code examples, local references and scoped diffs. No code, dependency, dataset or generated-artifact changes; no code checks rerun. The pre-existing `AD main.py` state was preserved.
