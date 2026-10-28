"""Read-only repository path guards; allocation is a separate responsibility."""

from pathlib import Path


class PathValidationError(ValueError):
    """A configured source or output violates the repository boundary."""


def resolve_cli_path(path: str | Path, repository_root: Path) -> Path:
    """Resolve CLI paths from the repository, allowing external prediction input."""
    candidate = Path(path)
    return (repository_root / candidate).resolve()


def repository_path(value: str, repository_root: Path, field: str) -> Path:
    """Return a canonical relative path, rejecting resolved repository escapes."""
    root = repository_root.resolve()
    try:
        return resolve_cli_path(value, root).relative_to(root)
    except (ValueError, OSError, RuntimeError) as exc:
        raise PathValidationError(
            f"{field}: path must resolve inside repository"
        ) from exc


def reject_symlink_ancestors(path: Path, root: Path) -> None:
    """Reject symlinks from the given repository root through the path itself."""
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise PathValidationError(f"Symlinked path is forbidden: {candidate}")


def validate_output_path(path: str | Path, repository_root: Path) -> Path:
    """Validate, without creating, a fresh output in a prescribed artifact tree."""
    root = repository_root.resolve()
    candidate = root / path
    reject_symlink_ancestors(candidate, root)
    resolved = candidate.resolve()
    roots = [
        root / name for name in ("data/interim", "data/processed", "models", "reports")
    ]
    if not any(
        resolved.is_relative_to(parent) and resolved != parent for parent in roots
    ):
        raise PathValidationError(f"Output is outside artifact locations: {path}")
    if resolved.is_relative_to((root / "data/raw").resolve()):
        raise PathValidationError(f"Output resolves into raw inputs: {path}")
    if candidate.exists() or candidate.is_symlink():
        raise PathValidationError(f"Output already exists: {path}")
    return resolved
