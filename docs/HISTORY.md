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

## Verification Record

M1–M3 code checks: 2026-10-29, CPython 3.14.6/macOS arm64. These historical results were not rerun for the documentation compaction on 2026-10-30; references, task state and scoped diffs were checked. New code verification is recorded in [M4 task 1](#m4-task-1--literature-reader-and-ud-adapter).

| Check | Command | Recorded result |
|---|---|---|
| Lint | `uv run ruff check .` | Failed: unchanged `scripts/fetch_dictionary.py` I001; changed-code lint passed |
| Formatting | `uv run ruff format --check .` | Failed: unchanged `scripts/fetch_dictionary.py` requires formatting; changed-code formatting passed |
| Types | `uv run mypy src/crf/` | Passed (2026-10-29): 19 source files |
| Tests | `uv run pytest tests/ -v` | Passed (2026-10-29): 594 tests, including 110 feature tests, 73 normalization/record cases, 89 shared label-space tests, 59 gold parser/audit tests and two offline integration tests |

## Recorded Limitations

- Runtime imports, `uv lock --check`/`uv pip check`, isolated editable imports, wheel/sdist contents and artifact ignore rules were verified during M1–M2. Tiny offline CRF fitting/marginal/joblib checks passed; cross-platform bitwise reproducibility is not claimed.
- Lint/format exception: unchanged `scripts/fetch_dictionary.py` fails I001 and formatting, reproduced before/after implementation. Changed-code checks pass; no checks were weakened. Existing `AD main.py` state remains untouched.
- Official gold availability/release identity, source separation and tokenizer compatibility remain unverified. M4 must handle upstream standalone formatting-character omissions, UD boundaries and aligned normalization with bounded paragraph streaming. No production training or final evaluation has run.

## Documentation Maintenance

- **2026-10-30 — Report compaction:** shortened milestone narratives/tables in ARCH/PLAN; verified all 34 checklist entries, pending milestone/corpus sections, architecture tables/code fences and local links. No code checks were rerun.
- **2026-10-30 — Documentation separation:** moved build/verification records here; kept architecture design in ARCH, goals/status in PLAN and placement rules in AGENTS. Verified migrated evidence, goal statuses, design tables/code examples, local references and scoped diffs. No code, dependency, dataset or generated-artifact changes; no code checks rerun. The pre-existing `AD main.py` state was preserved.
