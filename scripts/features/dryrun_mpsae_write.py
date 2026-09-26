"""Dry run: write MP-SAE feature scores into a copy of one slide store.

Runs the same call the MP-SAE notebook makes on data/*.zarr, but on a copy
of one store, and checks that:
  1. the original store is byte-identical before and after (SHA-256 of every file);
  2. the copy's existing tile table content (UNI_SAE_* block, obs, obsm, uns)
     is unchanged;
  3. the new UNI_MPSAE_* block has the expected shape and equals
     mpsae.transform(obsm["UNI_embedding"]);
  4. a second call leaves the copy untouched (skip-if-cached);
  5. non-table elements of the copy (shapes) are unchanged.

Usage:
    python dryrun_mpsae_write.py <output_dir> [slide_id]
"""

from fmsae_repro import DATA, MODELS
import hashlib
import os
import shutil
import sys
import time

import anndata as ad
import ezslide
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

import mesoslide as ms

MPSAE_PATH = MODELS / "MPSAE" / "mpsae_model.joblib"
TILE_KEY = "tiles"
POOLED_KEY = "UNI_embedding"
OLD_KEY = "UNI_SAE"
NEW_KEY = "UNI_MPSAE"
TABLE_REL = os.path.join("tables", "tiles_table")


def sha256_manifest(root):
    """{relative path: sha256} for every file under `root`."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[os.path.relpath(path, root)] = h.hexdigest()
    return out


def read_table(store):
    return ad.read_zarr(os.path.join(store, TABLE_REL))


def prefix_block(table, prefix):
    names = [v for v in table.var_names if v.startswith(f"{prefix}_")]
    return names, sp.csr_matrix(table[:, names].X)


def same_csr(a, b):
    a, b = sp.csr_matrix(a), sp.csr_matrix(b)
    a.sort_indices(); b.sort_indices()
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices) and np.array_equal(a.data, b.data))


def run_feature_extraction(store, slide_id, mpsae, device):
    slides = ezslide.open_slides(pd.DataFrame({"slide_id": [slide_id], "store": [store]}),
                                 attach_images=False)
    for _, slide in slides.items():
        ms.tl.feature_extraction(
            slide,
            model="uni",
            tile_key=TILE_KEY,
            key_added=POOLED_KEY,
            sparse=True,
            sparse_transform=lambda X: mpsae.transform(X, device=device),
            sparse_key_added=NEW_KEY,
            save=True,
        )
        slide.close()


def main():
    out_dir = sys.argv[1]
    slide_id = sys.argv[2] if len(sys.argv) > 2 else "LSP24521"
    original = str(DATA / f"{slide_id}.zarr")
    copy = os.path.join(out_dir, f"{slide_id}.zarr")
    device = "cuda"
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    print(f"Original: {original}\nCopy:     {copy}")
    orig_hash_before = sha256_manifest(original)
    orig_table = read_table(original)
    old_names, old_block = prefix_block(orig_table, OLD_KEY)
    print(f"Original table: {orig_table.shape}, {OLD_KEY} cols = {len(old_names)}, nnz = {old_block.nnz}")

    if os.path.exists(copy):
        raise FileExistsError(f"{copy} exists; remove it or choose another output dir")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copytree(original, copy)
    check("copy is byte-identical to original", sha256_manifest(copy) == orig_hash_before)

    mpsae = joblib.load(MPSAE_PATH)

    # First write
    t0 = time.time()
    run_feature_extraction(copy, slide_id, mpsae, device)
    elapsed = time.time() - t0
    print(f"feature_extraction on copy: {elapsed:.1f} s for {orig_table.n_obs} tiles")

    new_table = read_table(copy)
    names_after, old_block_after = prefix_block(new_table, OLD_KEY)
    check(f"{OLD_KEY} var names unchanged", names_after == old_names)
    check(f"{OLD_KEY} block bit-identical", same_csr(old_block, old_block_after))

    new_names, new_block = prefix_block(new_table, NEW_KEY)
    expected_names = [f"{NEW_KEY}_{i}" for i in range(mpsae.embed_dim_)]
    check(f"{NEW_KEY} var names 0..{mpsae.embed_dim_ - 1}", new_names == expected_names)
    check(f"{NEW_KEY} block has n_obs rows", new_block.shape[0] == orig_table.n_obs)
    ref = mpsae.transform(np.asarray(orig_table.obsm[POOLED_KEY], dtype=np.float32),
                          device=device, progress_bar=False)
    check(f"{NEW_KEY} block equals mpsae.transform(obsm)", same_csr(new_block, ref),
          f"nnz = {new_block.nnz}, mean L0 = {new_block.nnz / new_block.shape[0]:.1f}")

    check("obs unchanged", new_table.obs.equals(orig_table.obs))
    check("obs_names unchanged", list(new_table.obs_names) == list(orig_table.obs_names))
    check("obsm keys unchanged", sorted(new_table.obsm) == sorted(orig_table.obsm))
    for key in orig_table.obsm:
        a, b = orig_table.obsm[key], new_table.obsm[key]
        a = a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)
        b = b.to_numpy() if hasattr(b, "to_numpy") else np.asarray(b)
        check(f"obsm['{key}'] unchanged", a.shape == b.shape and np.array_equal(a, b))
    check("uns keys preserved", set(orig_table.uns) <= set(new_table.uns),
          f"added: {sorted(set(new_table.uns) - set(orig_table.uns))}")

    copy_hash_after_first = sha256_manifest(copy)
    changed = sorted(p for p in copy_hash_after_first
                     if orig_hash_before.get(p) != copy_hash_after_first[p])
    outside_table = [p for p in changed if not p.startswith(TABLE_REL) and p != "zarr.json"]
    check("no files outside the tile table changed in copy (except root zarr.json)",
          not outside_table, f"{outside_table[:5]}")
    removed = sorted(set(orig_hash_before) - set(copy_hash_after_first))
    removed_outside = [p for p in removed if not p.startswith(TABLE_REL)]
    check("no files outside the tile table removed", not removed_outside, f"{removed_outside[:5]}")

    # Second write must be a no-op
    t0 = time.time()
    run_feature_extraction(copy, slide_id, mpsae, device)
    check("second call leaves copy byte-identical (skip-if-cached)",
          sha256_manifest(copy) == copy_hash_after_first, f"{time.time() - t0:.1f} s")

    check("ORIGINAL store byte-identical after dry run", sha256_manifest(original) == orig_hash_before)

    size = lambda p: sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    print(f"Store size: original {size(original) / 1e6:.0f} MB -> copy {size(copy) / 1e6:.0f} MB")

    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
