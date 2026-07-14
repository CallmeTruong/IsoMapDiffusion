"""Helper functions for the isometric pipeline."""

from pathlib import Path


def resolve_path(path_str: str, project_root: Path = None) -> Path:
    """Resolve a relative path from project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if project_root is None:
        project_root = Path(__file__).parent.parent.resolve()
    return project_root / p


def get_project_root() -> Path:
    """Get project root: parent of src/"""
    return Path(__file__).parent.parent.resolve()


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
