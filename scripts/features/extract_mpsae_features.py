"""Add MP-SAE feature scores (UNI_MPSAE_*) to every slide's tile table.

Same call as the MP-SAE notebook and dryrun_mpsae_write.py, over the whole
manifest. Stores that already have UNI_MPSAE_* columns are skipped, so the
script can be re-run or split across nodes with --shard/--num-shards.

Refuses to run unless --backup-dir holds a SHA256SUMS file covering every
store's tile table (written by the pre-write backup).

Usage:
    python extract_mpsae_features.py --backup-dir DIR [--device cuda] [--shard 0 --num-shards 1]
"""

from fmsae_repro import MODELS, load_manifest
import argparse
import os
import time

import ezslide
import joblib
import pandas as pd

import mesoslide as ms

MPSAE_PATH = MODELS / "MPSAE" / "mpsae_model.joblib"
TILE_KEY = "tiles"
POOLED_KEY = "UNI_embedding"
SPARSE_KEY = "UNI_MPSAE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    manifest = load_manifest()
    sums = os.path.join(args.backup_dir, "SHA256SUMS")
    backed_up = {line.split()[1].split("/")[0] for line in open(sums)}
    missing = [s for s in manifest["store"] if os.path.basename(s) not in backed_up]
    if missing:
        raise RuntimeError(f"No backup for {len(missing)} stores, e.g. {missing[:3]}")

    shard = manifest.iloc[args.shard::args.num_shards].reset_index(drop=True)
    mpsae = joblib.load(MPSAE_PATH)
    print(f"Shard {args.shard}/{args.num_shards}: {len(shard)} slides on {args.device}", flush=True)

    for i, row in shard.iterrows():
        t0 = time.time()
        slides = ezslide.open_slides(pd.DataFrame([row]), attach_images=False)
        for slide_id, slide in slides.items():
            ms.tl.feature_extraction(
                slide,
                model="uni",
                tile_key=TILE_KEY,
                key_added=POOLED_KEY,
                sparse=True,
                sparse_transform=lambda X: mpsae.transform(X, device=args.device, progress_bar=False),
                sparse_key_added=SPARSE_KEY,
                save=True,
            )
            table = slide.tables[ms.tile_table_key(TILE_KEY)]
            n_new = sum(v.startswith(f"{SPARSE_KEY}_") for v in table.var_names)
            print(f"[{i + 1}/{len(shard)}] {slide_id}: {table.n_obs} tiles, {n_new} {SPARSE_KEY} cols, "
                  f"X {table.X.shape}, {time.time() - t0:.0f} s", flush=True)
            slide.close()


if __name__ == "__main__":
    main()
