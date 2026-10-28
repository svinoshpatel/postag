# Agent Entrypoint and Operational Rules

This repository implements the Ukrainian Semi-Supervised CRF POS Tagger.

## 1. Required Context and Document Authority

1. Before implementing or reviewing changes, read [docs/ARCH.md](docs/ARCH.md) for technical specifications and [PLAN.md](PLAN.md) for task state.
2. Treat `AGENTS.md` as the authority for repository operating procedures, `docs/ARCH.md` for technical requirements, and `PLAN.md` for task tracking. These files do not override explicit user instructions or higher-priority agent instructions.
3. Supporting documentation must conform to the applicable operational and technical requirements. Do not infer an exception from an example or an existing implementation.
4. Do not duplicate technical specifications or task checklists in this file. Reference their authoritative location.
5. When applicable technical requirements conflict, the agent may resolve the conflict autonomously. Before implementing affected behavior, document the conflicting requirements, chosen interpretation, rationale, and consequences in `docs/ARCH.md`. Update superseded statements so the specification remains internally consistent.
6. When the current repository differs from the prescribed architecture, the agent may choose an adaptation or migration necessary for the authorized task. Document the rationale and compatibility implications before implementation. Do not undertake unrelated migrations.
7. This autonomy does not expand the user-authorized scope or override explicit restrictions. If documentation edits are prohibited, report the decision and rationale without modifying files.

## 2. Execution Standards and Action Boundaries

1. During implementation, the agent may inspect files, edit task-relevant code and documentation, and run local verification without additional confirmation.
2. Explicit read-only or no-execution instructions take precedence. In read-only reviews, do not update `PLAN.md`, generate artifacts, or run commands that write files. If execution is prohibited, use static inspection only.
3. Before executing a command, identify its expected filesystem, network, and external-service effects. An unfamiliar script must be inspected before execution.
4. Pipeline entry points must reside under `scripts/`, use `argparse` or `click`, and delegate reusable logic to the core package designated by `docs/ARCH.md`. Existing entry points may remain during a documented migration.
5. Notebooks may contain exploration and visualization code. Reusable production logic must reside in the core package.
6. Execute commands from the repository root unless a command explicitly requires another working directory.
7. The agent may install task-required dependencies in the project environment and download task-required resources without additional approval. Training runs are permitted when training is part of the requested task.
8. Explicit authorization is required before using paid services, uploading project content to external services, changing system-wide configuration, or overwriting existing generated artifacts. Existing authorization for the specific action satisfies this requirement. Original datasets remain immutable.
9. Git operations follow the separate Git Protocol below.

## 3. Coding Guidelines

1. New or modified functions must have parameter and return annotations; `self` and `cls` need not be annotated. Annotation syntax must support the selected Python version documented in `docs/ARCH.md`.
2. Use dataclasses or named tuples for records with a fixed schema. Use mappings for dynamic feature dictionaries. Prefer immutable records; document why mutation is necessary when choosing mutable records.
3. Use `pathlib.Path` for filesystem paths. Do not hard-code machine-specific absolute paths. Runtime-resolved absolute paths and explicitly supplied input/output paths are permitted.
4. Use module-level logging for diagnostics, such as `logger = logging.getLogger(__name__)`. CLI result data may be written to stdout; diagnostics must go to stderr. Do not mix diagnostic messages with machine-readable output.
5. Specify text encoding explicitly. Default to UTF-8; permit explicitly configured source encodings when required by the data specification. Generated text artifacts must use UTF-8.
6. Follow the repository's Ruff configuration. Formatting and lint verification are separate required checks. The agent may install and configure Ruff in the project environment when needed. Keep configuration in the project's standard configuration file and document material rule choices. Do not assume Ruff is already available.
7. Prefer a newer stable Python release supported by the project's dependencies. The agent may upgrade Python and update dependency declarations, environment configuration, and `docs/ARCH.md` within the task's scope. Document the rationale and verify compatibility before declaring the upgrade complete. Historical Python versions are not immutable requirements. Generated syntax must match the selected, documented Python version.
8. Apply these standards to code changed for the task. Do not perform unrelated repository-wide cleanup.

## 4. Verification and Test Generation

### Three-Stage Protocol

1. **Implement:** Identify applicable requirements and expected behavior, then implement one coherent change.
2. **Review against the specification:** Before writing new tests, record a concise mapping of requirements to implemented behavior. Identify boundary conditions and failure cases. Resolve discrepancies before proceeding. This self-review is not proof of correctness.
3. **Test and verify:** Write tests with expected results derived from the specification, covering normal behavior, boundaries, and failures. Run focused tests during development and the full required checks before marking the task complete.

Unit tests must use small, deterministic fixtures and must not require network access or production datasets. Shared fixtures belong in `tests/conftest.py`; fixtures used by one test module may remain local. Integration tests must be separately identifiable and document their dependencies.

### Required Checks

Run verification from the repository root through `uv`:

| Check      | Command                                      |
|------------|----------------------------------------------|
| Lint       | `uv run ruff check .`                        |
| Formatting | `uv run ruff format --check .`               |
| Types      | `uv run mypy <documented-core-package-path>` |
| Tests      | `uv run pytest tests/ -v`                    |

The type-checking command is a template, not a literal shell command. Substitute the actual core package path before execution. The agent must establish missing tooling and replace this placeholder with the actual package path when configuring the project. Record each check's result. Never report an unavailable, skipped, or failing check as passed.

Pre-existing failures outside the task's scope do not block completion when the agent provides evidence that they are unrelated and verifies the changed behavior. Record the failing checks, supporting evidence, and remaining limitations in `PLAN.md` and the final report. Do not weaken checks or modify unrelated code merely to obtain a passing result.

If a failure prevents verification of the changed behavior, keep the task incomplete and report the blocker. Documentation-only and read-only tasks follow the applicability rules below.

## 5. Task Execution and Tracking

1. For multi-step implementation tasks, read `PLAN.md` and add task-relevant steps before editing code. Preserve unrelated entries.
2. Work in coherent increments. Verify each increment before beginning work that depends on it. Independent inspection and analysis may proceed concurrently.
3. Mark a step `[x]` only when its acceptance criteria are met and applicable verification is complete. Document any permitted pre-existing verification failures beside the completed entry.
4. Run focused checks between increments and all four required checks before completing a code-changing task, subject to the documented pre-existing-failure exception above.
5. For documentation-only changes, verify references, consistency, and any affected command examples by inspection. Code verification is required only when executable behavior or configuration changes.
6. Do not edit `PLAN.md` during read-only reviews. For a single-step change, update an existing relevant entry if present; creating a new tracking entry is optional.
7. Report completed work, verification results, and unresolved limitations without claiming unperformed checks.

## 6. Git Protocol

1. The agent may run read-only Git inspection commands, including `git status`, `git diff`, `git log`, and `git show`.
2. Git operations that change the index, working tree, history, repository configuration, or remotes require explicit user authorization. This includes staging, committing, switching branches, stashing, resetting, cleaning, merging, rebasing, fetching, and pushing. Authorization applies only to the requested operation and scope.
3. After completing a task with file changes, provide a suggested commit message and commands naming the actual changed paths. Do not recommend staging unrelated changes. Omit commit recommendations for tasks with no changes.

## 7. Error Recovery

1. Inspect the failure output and identify whether the failure originates in the changed code, existing code, dependencies, or environment.
2. Reproduce the failure using the smallest relevant check within the authorized execution scope. Inspect potential side effects before running a reproduction.
3. State the suspected cause and supporting evidence before applying a targeted repair. Distinguish a hypothesis from a confirmed cause.
4. The agent may repair task-related implementation defects without additional approval, including defects in existing core logic. Existing action restrictions remain applicable during recovery.
5. Preserve public interface compatibility where practical. A necessary breaking change requires a documented rationale, updates to affected callers and tests, and disclosure in the final report.
6. After repair, rerun the failing check and relevant regression checks. Run the full required verification before task completion.
7. Do not repeat an unsuccessful repair without new evidence. If recovery requires an action outside the authorized scope, report the blocker and request the specific authorization needed.

## 8. Data and Model Artifacts

1. Original inputs under `data/raw/`, including annotations, CoNLL-U files, and literary text, must not be modified, deleted, renamed, or overwritten. Transformations must produce separate derived artifacts.
2. Artifact locations are defined exclusively in `docs/ARCH.md`. Before writing outputs, ensure their resolved paths cannot overwrite original inputs.
3. Before producing artifacts, inspect `.gitignore` and add missing exclusions for generated datasets, model binaries, weights, and serialization formats used by the project. Small, intentional test fixtures may remain versioned.
4. Existing generated artifacts must not be overwritten without explicit authorization. Otherwise, use a distinct output location.
5. Dataset and model artifacts must not be staged or committed, except for the intentional test fixtures allowed above. Before recommending a commit, inspect the proposed paths for prohibited artifacts.

## 9. Environment and Reproducibility

1. Manage project dependencies and verification tools through `uv` in a project-local environment. Maintain dependency declarations and a lockfile so the environment can be reproduced.
2. Scripts that perform sampling, randomized feature generation, or training must accept `--seed`, use a documented default, and propagate it to every applicable random-number generator or library setting.
3. Do not assume that setting a seed guarantees deterministic results. Document any remaining nondeterminism and its effect on verification.
4. Training and evaluation runs must record the effective configuration, seed, Python and dependency versions, and input dataset identifiers alongside their outputs.
5. Reproducibility metadata must not modify original datasets or include credentials.
