"""Add SAE-derived sparse feature scores to every slide's tile table.

Iterates over every zarr store in DATA_DIR (written by preprocess_wsi.py,
which already holds tiles and the pooled UNI_embedding) and calls
mesoslide.tl.feature_extraction(sparse=True, sparse_transform=sae.transform)
to write the reference SAE's sparse feature matrix into table.X. Each store
is skip-if-cached (feature_extraction only (re)computes a var-name prefix
that isn't already present), so re-running the whole script is safe and
only does work for stores that don't have it yet.

Loading the reference checkpoint (ref_SAE.joblib) needs a compatibility
shim for two reasons:
  1. It was pickled under the pre-rename package name
     `meson.tools.embedders.SAE`, which no longer exists after the
     meson -> mesoslide rename.
  2. Its class's own transform() (mesoslide.tools.embedders._legacy.SAE,
     historical variants) call sklearn APIs (BaseEstimator._validate_data,
     or attributes like is_pruned_/active_features_ from a later pruning
     feature) that either no longer exist in the installed sklearn or were
     never set on this particular checkpoint.
Rather than depend on one specific historical source file, this defines
minimal SimpleAutoencoder/SparseAutoencoder classes under a synthetic
meson.tools.embedders.SAE module. Unpickling restores an object's __dict__
directly and never calls __init__, so these classes only need to declare
the same submodule names (encoder/decoder) and implement transform() against
the checkpoint's actual fitted attributes (model_, scale_factor_,
embed_dim_, batch_size), confirmed by inspecting the checkpoint directly.

Usage:
    uv run python extract_sae_features.py
"""

from fmsae_repro import DATA, LEGACY_ANALYSIS
import os
import sys
import types
from glob import glob

import ezslide
import joblib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset

import mesoslide as ms

SAE_PATH = f'{LEGACY_ANALYSIS}/TB/3D_HnE/ref_SAE.joblib'
DATA_DIR = str(DATA)

TILE_KEY = 'tiles'
POOLED_KEY = 'UNI_embedding'
SPARSE_KEY = 'UNI_SAE'


class _SimpleAutoencoder(nn.Module):
    """Pickle shim for the checkpoint's inner torch module.

    __init__'s arguments are never used at unpickling time: pickling an
    nn.Module restores its state directly into __dict__ (encoder/decoder
    submodules and their weights included) without ever calling __init__.
    This only has to declare the same submodule names for that state to
    attach correctly.
    """

    def __init__(self, input_dim=1, expansion_factor=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * expansion_factor)
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)

    def forward(self, x):
        h = F.relu(self.encoder(x))
        return self.decoder(h), h


class _SparseAutoencoder(TransformerMixin, BaseEstimator):
    """Pickle shim reproducing ref_SAE.joblib's fitted transform().

    Scales the input by the fitted scale_factor_, runs it through the
    inner autoencoder, and keeps only the nonzero (post-ReLU) hidden
    activations as a sparse matrix -- the same contract every historical
    version of mesoslide's SAE embedder used, minus API calls (sklearn
    input validation, feature-pruning bookkeeping) this checkpoint doesn't
    need.
    """

    def transform(self, X, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)
        X_scaled = torch.as_tensor(np.asarray(X, dtype=np.float32)) * self.scale_factor_
        loader = DataLoader(TensorDataset(X_scaled), batch_size=self.batch_size, shuffle=False)

        rows, cols, vals = [], [], []
        with torch.no_grad():
            for idx, (batch,) in enumerate(loader):
                _, hidden = self.model_.forward(batch.to(device))
                sparse = hidden.to_sparse().cpu()
                row_ind, col_ind = sparse.indices().numpy()
                rows.append(row_ind + idx * self.batch_size)
                cols.append(col_ind)
                vals.append(sparse.values().numpy())

        data = np.concatenate(vals)
        row_ind, col_ind = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix((data, (row_ind, col_ind)), shape=(X.shape[0], self.embed_dim_))


def load_sae(path):
    """Load ref_SAE.joblib via the meson.tools.embedders.SAE pickle shim."""
    sae_module = types.ModuleType("meson.tools.embedders.SAE")
    sae_module.SimpleAutoencoder = _SimpleAutoencoder
    sae_module.SparseAutoencoder = _SparseAutoencoder
    for name in ("meson", "meson.tools", "meson.tools.embedders"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["meson.tools.embedders.SAE"] = sae_module
    return joblib.load(path)


def main():
    sae = load_sae(SAE_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stores = sorted(glob(os.path.join(DATA_DIR, "*.zarr")))
    print(f"Found {len(stores)} zarr stores in {DATA_DIR}")

    for store in stores:
        lsp_id = os.path.basename(store).split(".")[0]
        wsi = ezslide.read_wsi(store, attach_images=True)

        ms.tl.feature_extraction(
            wsi,
            model="uni",  # only resolved/loaded if UNI_embedding isn't already cached
            tile_key=TILE_KEY,
            key_added=POOLED_KEY,
            sparse=True,
            sparse_transform=lambda pooled: sae.transform(pooled, device=device),
            sparse_key_added=SPARSE_KEY,
            save=True,
        )

        table = wsi.tables[ms.tile_table_key(TILE_KEY)]
        print(f"{lsp_id}: X shape {table.X.shape}, nnz {table.X.nnz}")


if __name__ == "__main__":
    main()
