from fmsae_repro import MODELS, load_manifest
import ezslide
import lazyslide as zs
import mesoslide as ms
import pandas as pd

# ── sweep setup ──────────────────────────────────────────────────────────────
import hashlib
import json
from dataclasses import dataclass, asdict, replace
from glob import glob
from pathlib import Path

import joblib
from mesoslide.tools.sae import SparseAutoencoder

RUNS_DIR = Path(f"{MODELS}/SAE_sweep")
DEVICE = "cuda:1"


@dataclass(frozen=True)
class SAERunConfig:
    fraction: float
    num_steps: int
    target_sparsity: float
    expansion_factor: int
    learning_rate: float
    random_state: int
    batch_size: int = 2048
    min_lambda: float = 1e-6
    max_lambda: float = 5

def run_id(config: SAERunConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


baseline = SAERunConfig(
    fraction=1.0,
    num_steps=2000,
    target_sparsity=0.001,
    expansion_factor=64,
    learning_rate=5e-5,
    random_state=0,
)

scans = {
    "fraction": [0.01, 0.03, 0.1, 0.3, 1.0],
    "num_steps": [500, 1000, 2000, 5000],
    "target_sparsity": [1e-4, 5e-4, 1e-3, 5e-3],
    "expansion_factor": [16, 32, 64],
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "random_state": [0, 1, 2],
}

configs = {baseline}
for field, values in scans.items():
    for v in values:
        configs.add(replace(baseline, **{field: v}))

print(f"{len(configs)} configs to train (1 baseline + {sum(len(v) for v in scans.values())} scan points, deduped)")

manifest = load_manifest()
slides = ezslide.open_slides(manifest, attach_images=False)
print("Slides loaded")

# ── train ──────────────────────────────────────────────────────────────────
for config in configs:
    out_dir = RUNS_DIR / run_id(config)
    if (out_dir / "sae.joblib").exists():
        print(f"skip (already trained): {out_dir.name}")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"training: {out_dir.name} -- {asdict(config)}")

    sae = SparseAutoencoder(
        expansion_factor=config.expansion_factor,
        batch_size=config.batch_size,
        num_steps=config.num_steps,
        min_lambda=config.min_lambda,
        max_lambda=config.max_lambda,
        target_sparsity=config.target_sparsity,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
    )
    sae.fit(
        slides,
        obsm_key="UNI_embedding",
        tile_key="tiles",
        fraction=config.fraction,
        device=DEVICE,
        verbose=0,
    )
    joblib.dump(sae, out_dir / "sae.joblib")

# ── later: build an index to load and assess runs one by one ───────────────
# rows = []
# for config_path in sorted(glob(str(RUNS_DIR / "*/config.json"))):
#     run_dir = Path(config_path).parent
#     rows.append({**json.load(open(config_path)), "run_dir": run_dir})
# runs_df = pd.DataFrame(rows)
# sae = joblib.load(runs_df.loc[0, "run_dir"] / "sae.joblib")
