"""Project paths, all derived from the repository root.

No absolute path is hardcoded in the notebooks or scripts: they import these
constants instead. Locations outside the repository (raw WSIs, earlier analysis
outputs) come from environment variables, which can also be set in an untracked
`.env` file at the repository root (see `.env.example`).
"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from `path` as environment variables (existing ones win)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# src/fmsae_repro/paths.py -> repository root (requires an editable install).
ROOT = Path(os.environ.get("FMSAE_ROOT", Path(__file__).resolve().parents[2]))
if not (ROOT / "pyproject.toml").is_file():
    raise RuntimeError(
        f"Repository root not found at {ROOT}. Install this package in editable mode "
        "(`uv pip install -e .`) or set FMSAE_ROOT."
    )
_load_dotenv(ROOT / ".env")

DATA = ROOT / "data"          # zarr stores and manifest.csv, written by scripts/preprocessing
MODELS = ROOT / "models"      # fitted SAE / LLC / DL models and token clusterers
RESULTS = ROOT / "results"    # tables, figures, patch collections
LOGS = ROOT / "logs"          # Slurm logs

# Outside the repository.
RAW_DATA = Path(os.environ.get("FMSAE_RAW_DATA", ROOT / "raw_data"))
LEGACY_ANALYSIS = Path(os.environ.get("FMSAE_LEGACY_ANALYSIS", ROOT / "external" / "analysis_meson"))
