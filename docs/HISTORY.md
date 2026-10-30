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

## Verification Record

Latest code checks: 2026-10-29, CPython 3.14.6/macOS arm64. Historical results below were not rerun for the documentation compaction on 2026-10-30; references, task state and scoped diffs were checked.

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
