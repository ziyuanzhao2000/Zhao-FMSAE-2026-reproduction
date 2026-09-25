"""Helpers shared by the reproduction notebooks and scripts."""
from .paths import DATA, LEGACY_ANALYSIS, LOGS, MODELS, RAW_DATA, RESULTS, ROOT


def load_manifest(name: str = "manifest.csv"):
    """Read `data/<name>` and resolve its `store` column (relative to the repo root) to full paths."""
    import pandas as pd

    manifest = pd.read_csv(DATA / name)
    manifest["store"] = [str(ROOT / store) for store in manifest["store"]]
    return manifest


__all__ = ["ROOT", "DATA", "MODELS", "RESULTS", "LOGS", "RAW_DATA", "LEGACY_ANALYSIS", "load_manifest"]
