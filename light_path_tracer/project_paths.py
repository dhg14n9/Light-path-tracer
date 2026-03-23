"""Project-wide filesystem locations for generated outputs."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ASYM_DATA_DIR = DATA_DIR / "asym_data"
OUTPUT_DIR = DATA_DIR / "outputs"


def default_output_path(filename: str) -> Path:
    """Return a standard path under data/outputs for one generated file."""
    return OUTPUT_DIR / str(filename)


def resolve_output_path(path_value: str | Path) -> Path:
    """Resolve a user-facing output path, defaulting bare filenames into data/outputs."""
    path = Path(path_value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return path
    return default_output_path(path.name)

