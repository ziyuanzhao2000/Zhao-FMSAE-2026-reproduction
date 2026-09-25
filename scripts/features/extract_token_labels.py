"""Extract per-token tissue-class label maps from mesoslide zarr stores.

Reuses the zarr stores written by preprocess_wsi.py (tiles and the pooled
UNI embedding are already computed and saved there) and adds a dense
per-token pass whose reducer is the same GPU-accelerated 1-NN classifier
used in mesoslide/scripts/extract_token_map_v6.py, via
mesoslide.tl.feature_extraction(dense=True, reducer=...). The pooled
embedding is left untouched: feature_extraction only (re)computes a key
that is missing from obsm.

Token predictions are assembled into one label map at level-0 resolution.
Each tile keeps only its central `stride_px / patch_px` tokens -- the ones
with the most surrounding context, matching extract_token_map_v6.py's
keep-tokens/margin scheme -- so tiles tile the output exactly, without
gaps or double writes.

Usage:
    export HF_TOKEN=hf_...
    uv run python extract_token_labels.py <array-index>
"""

from fmsae_repro import DATA, RESULTS, LEGACY_ANALYSIS
import os
import sys
from glob import glob

import ezslide
import joblib
import numpy as np
import tifffile
import torch

import mesoslide as ms

CLASSIFIER_PATH = (
    f'{LEGACY_ANALYSIS}/SAE_paper/notebooks/'
    'post-submission/knn_classifier_with_necrotic.joblib'
)
DATA_DIR = str(DATA)
OUTPUT_DIR = f'{RESULTS}/token_maps'

TILE_KEY = 'tiles'
POOLED_KEY = 'UNI_embedding'
LABEL_KEY = 'UNI_knn_labels'


def select_device():
    """Pick the CUDA device with the most free memory, else CPU."""
    if not torch.cuda.is_available():
        print('CUDA not available, using CPU.')
        return torch.device('cpu')

    best_device, max_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > max_free:
            max_free, best_device = free, i

    print(f'Using CUDA device: {best_device} with {max_free / 1024**2:.2f} MB free.')
    return torch.device(f'cuda:{best_device}')


class GpuKNeighborsClassifier:
    """GPU re-implementation of a fitted cosine-similarity KNeighborsClassifier.

    Same as the one in mesoslide/scripts/extract_token_map_v6.py.
    """

    def __init__(self, sk, device):
        ref = torch.as_tensor(np.ascontiguousarray(sk._fit_X), dtype=torch.float32, device=device)
        self.ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-30)
        self.y = torch.as_tensor(np.asarray(sk._y), device=device)
        self.classes = torch.as_tensor(sk.classes_.astype(np.int64), device=device)
        self.k = int(sk.n_neighbors)
        self.chunk = max(1024, int(2e8 // len(ref)))

    @torch.inference_mode()
    def predict(self, x):
        out = []
        for xb in x.split(self.chunk):
            xn = xb / xb.norm(dim=1, keepdim=True).clamp_min(1e-30)
            idx = (xn @ self.ref.T).topk(self.k, dim=1).indices  # largest sim == nearest
            out.append(torch.mode(self.y[idx], dim=1).values)
        return self.classes[torch.cat(out)].to(torch.uint8)


class KnnTokenReducer:
    """feature_extraction(dense=True) reducer: per-token 1-NN class label.

    Records `grid_side` (tokens per tile side) from the first batch, since
    the map-assembly step needs it and the model's token grid is otherwise
    not exposed to this script.
    """

    def __init__(self, gpu_knn):
        self.gpu_knn = gpu_knn
        self.grid_side = None

    def __call__(self, patch_tokens):
        b, n, d = patch_tokens.shape
        if self.grid_side is None:
            self.grid_side = int(round(n ** 0.5))
            if self.grid_side ** 2 != n:
                raise RuntimeError(f'Expected a square token grid, got {n} tokens.')
        labels = self.gpu_knn.predict(patch_tokens.reshape(b * n, d))
        return labels.reshape(b, n).float()


def assemble_label_map(table, grid_side, tile_px, stride_px):
    """Tile per-tile token-label grids into one level-0 label map.

    Each tile keeps only its central `keep x keep` tokens
    (keep = stride_px // patch_px); adjacent tiles are spaced exactly
    `keep` tokens apart, so the kept regions cover the slide without gaps
    or overlap -- the same keep-tokens scheme as extract_token_map_v6.py.
    """
    patch_px = tile_px // grid_side
    keep = stride_px // patch_px
    margin = (grid_side - keep) // 2
    if margin * 2 + keep != grid_side or stride_px % patch_px != 0:
        raise RuntimeError(
            f'tile_px={tile_px}, stride_px={stride_px}, grid_side={grid_side} '
            'do not divide into a centered keep-tokens crop.'
        )

    x = table.obs['x'].to_numpy()
    y = table.obs['y'].to_numpy()
    labels = table.obsm[LABEL_KEY].astype(np.uint8)

    tok_row = y // patch_px + margin
    tok_col = x // patch_px + margin
    n_rows = int(tok_row.max()) + keep
    n_cols = int(tok_col.max()) + keep

    token_map = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for i in range(len(table)):
        grid = labels[i].reshape(grid_side, grid_side)
        cropped = grid[margin:margin + keep, margin:margin + keep]
        r, c = int(tok_row[i]), int(tok_col[i])
        token_map[r:r + keep, c:c + keep] = cropped

    # Nearest-neighbour upsample: each token cell expands to its patch_px x patch_px footprint.
    return np.repeat(np.repeat(token_map, patch_px, axis=0), patch_px, axis=1)


def main():
    stores = sorted(glob(os.path.join(DATA_DIR, '*.zarr')))
    store = stores[int(sys.argv[1])]
    lsp_id = os.path.basename(store).split('.')[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{lsp_id}_token_labels.ome.tif')
    if os.path.exists(output_path):
        print(f'Output already exists at {output_path}, skipping.')
        return

    print(f'Processing {lsp_id} from {store}')
    wsi = ezslide.read_wsi(store, attach_images=True)

    device = select_device()
    sk_clf = joblib.load(CLASSIFIER_PATH)
    gpu_knn = GpuKNeighborsClassifier(sk_clf, device)
    reducer = KnnTokenReducer(gpu_knn)

    wsi = ms.tl.feature_extraction(
        wsi,
        model='uni',
        tile_key=TILE_KEY,
        key_added=POOLED_KEY,
        dense=True,
        reducer=reducer,
        dense_key_added=LABEL_KEY,
        token=os.environ.get('HF_TOKEN'),
        batch_size=128,
        device=str(device),
        num_workers=1,
        save=True, 
    )

    # table = wsi.tables[ms.tile_table_key(TILE_KEY)]
    # spec = wsi.tile_spec(TILE_KEY)
    # tile_px = int(spec.base_height)
    # stride_px = int(spec.base_stride_height)

    # token_map = assemble_label_map(table, reducer.grid_side, tile_px, stride_px)
    # print(f'Token label map: {token_map.shape} ({token_map.nbytes / 1e9:.2f} GB)')

    # tmp_path = output_path + '.tmp'
    # tifffile.imwrite(
    #     tmp_path, token_map,
    #     ome=True, dtype='uint8', bigtiff=True,
    #     photometric='minisblack',
    #     tile=(1024, 1024),
    #     compression='zstd',
    #     compressionargs={'level': 1},
    #     maxworkers=os.cpu_count(),
    # )
    # os.replace(tmp_path, output_path)
    # print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
