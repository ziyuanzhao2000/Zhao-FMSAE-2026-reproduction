"""Reconstruction FVU of the reference SAE (SAE1) and the dictionary-learning model over all patches.

FVU = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2, each model in its own scaled input
space (FVU is invariant to the input scale). SAE reconstructions are computed two ways:
(a) decoding the stored scores in tiles_table.X, (b) the model's own forward pass.
"""
import socket

import joblib
import numpy as np
import torch
import ezslide

from fmsae_repro import MODELS, load_manifest

dev = "cuda"
print("node:", socket.gethostname(), "| GPU:", torch.cuda.get_device_name(0), flush=True)

sae = joblib.load(MODELS / "ref_SAE.joblib")
enc, dec = sae.model_.encoder.to(dev), sae.model_.decoder.to(dev)
s_sae = float(sae.scale_factor_)
dl = joblib.load(MODELS / "DL" / "dl_model.joblib")
D = torch.tensor(dl.components_, dtype=torch.float64, device=dev)
mean_dl = torch.tensor(dl.mean_, dtype=torch.float64, device=dev)
print(f"DL model: n_components={dl.n_components}, alpha={dl.alpha:.4f}, "
      f"n_steps={dl._training_log['n_steps']}", flush=True)

sae_names = [f"UNI_SAE_{i}" for i in range(sae.embed_dim_)]
manifest = load_manifest()
acc = {k: 0.0 for k in ("n", "sse_sae_stored", "sse_sae_forward", "sse_dl", "l0_sae", "l0_dl")}
sum_x = np.zeros(1024)
sum_sq = 0.0
max_code_diff = 0.0

for k, sid in enumerate(manifest.slide_id):
    t = ezslide.open_slides(manifest.iloc[[k]], attach_images=False)[sid]["tiles_table"]
    X = t.obsm["UNI_embedding"].astype(np.float64)
    n = len(X)
    sum_x += X.sum(0)
    sum_sq += float((X ** 2).sum())
    with torch.no_grad():
        # SAE (a): stored scores -> decoder, in the SAE's scaled space.
        H = t[:, sae_names].X.tocsr().astype(np.float32)
        acc["l0_sae"] += H.nnz
        H_t = torch.sparse_csr_tensor(torch.tensor(H.indptr, dtype=torch.int64),
                                      torch.tensor(H.indices, dtype=torch.int64),
                                      torch.tensor(H.data), size=H.shape).to(dev)
        Xs = torch.tensor(X, device=dev) * s_sae
        xhat_a = (H_t @ dec.weight.T.float()).double() + dec.bias.double()
        acc["sse_sae_stored"] += float(((Xs - xhat_a) ** 2).sum())
        # SAE (b): forward pass, in row blocks.
        for b in range(0, n, 8192):
            xb = Xs[b:b + 8192].float()
            h = torch.relu(enc(xb))
            xhat_b = dec(h).double()
            acc["sse_sae_forward"] += float(((Xs[b:b + 8192] - xhat_b) ** 2).sum())
            if b == 0:
                stored = torch.tensor(H[:len(xb)].toarray(), device=dev)
                max_code_diff = max(max_code_diff, float((h - stored).abs().max()))
        # DL, in its scaled space.
        C = dl.transform(t.obsm["UNI_embedding"], device=dev).tocsr()
        acc["l0_dl"] += C.nnz
        C_t = torch.tensor(C.toarray(), dtype=torch.float64, device=dev)
        Xd = torch.tensor(X, device=dev) * dl.scale_factor_
        acc["sse_dl"] += float(((Xd - (C_t @ D + mean_dl)) ** 2).sum())
    acc["n"] += n
    print(f"{sid}: {n:,} patches done", flush=True)

n = acc["n"]
mu = sum_x / n
sst_raw = sum_sq - n * float(mu @ mu)          # sum ||x - mean||^2 in raw units
sst_raw_zero = sum_sq
print(f"\npatches: {n:,}")
print(f"SAE stored-score vs forward-pass codes, first block per slide: max |diff| = {max_code_diff:.2e}")
for name, sse, s in (("SAE1 (stored scores)", acc["sse_sae_stored"], s_sae),
                     ("SAE1 (forward pass)", acc["sse_sae_forward"], s_sae),
                     ("DL (256 atoms)", acc["sse_dl"], dl.scale_factor_)):
    print(f"{name:22s} FVU vs mean {sse / (s * s * sst_raw):.4f} | FVU vs zero {sse / (s * s * sst_raw_zero):.4f}")
print(f"mean L0: SAE1 {acc['l0_sae'] / n:.2f} | DL {acc['l0_dl'] / n:.2f}")
