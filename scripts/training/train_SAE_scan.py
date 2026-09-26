"""Hyperparameter sensitivity scan for the SAE, centered on the SAE1-equivalent recipe.

Baseline: per-sample loss, input_norm='d', fixed lambda, with optimizer settings
rescaled so that training matches SAE1 (see
notebooks/training/train_SAE_per_sample_repro.ipynb). Each scan varies one
hyperparameter from the baseline; a joint scan varies expansion factor and lambda.

Every run is evaluated on all 40 slides and compared with the paper model
(ref_SAE.joblib, 361 selected features in reordered_active_features.csv).

Outputs in MODELS/SAE_sweep/:
    <run_id>/sae.joblib, config.json, metrics.json, stats.npz, feature_selection.png
    summary.csv, scan_summary.png, steps_by_slides.png, seed_pairwise.csv

Usage:
    python scripts/training/train_SAE_scan.py [--device cuda:1] [--dry-run] [--eval-only]
"""
import argparse
import functools
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields, replace

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ezslide
from fmsae_repro import LEGACY_ANALYSIS, MODELS, load_manifest
from mesoslide._slides import SlideSource
from mesoslide.tools.sparse_coding._sae import SparseAutoencoder

RUNS_DIR = MODELS / "SAE_sweep"
REF_MODEL = MODELS / "ref_SAE.joblib"
REF_FEATURES = LEGACY_ANALYSIS / "SAE_paper/TB/3D_HnE/relu_sae/reordered_active_features.csv"

# ── SAE1-equivalent recipe ───────────────────────────────────────────────────
D_EMB = 1024                 # UNI embedding dimension
C = D_EMB ** 0.25            # input scale ratio between input_norm='d' and 'sqrt_d'
B_SAE1 = 2048                # SAE1 batch size
# SAE1 adaptive-lambda floor (1e-6) rescaled to per-sample loss and input_norm='d';
# 1.13 compensates for the controller raising lambda during the first ~70 steps.
L1_BASELINE = round(1e-6 * B_SAE1 * D_EMB * C * 1.13, 4)
RECIPE = "sae1_equiv_v1"


@dataclass(frozen=True)
class SAERunConfig:
    n_slides: int = 40                 # training slides, evenly spaced across the 40
    num_steps: int = 2000
    l1_coefficient: float = L1_BASELINE
    expansion_factor: int = 64
    batch_size: int = 2048
    learning_rate: float = 5e-5
    random_state: int = 0
    recipe: str = RECIPE               # identifies the fixed settings in sae_kwargs()


def sae_kwargs(config: SAERunConfig) -> dict:
    """SparseAutoencoder arguments; the fixed settings make training match SAE1."""
    return dict(
        expansion_factor=config.expansion_factor,
        batch_size=config.batch_size,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        input_norm="d",
        loss_normalization="per_sample",
        lambda_mode="fixed",
        l1_coefficient=config.l1_coefficient,
        grad_clip_norm=None,
        adam_eps=1e-8 * D_EMB * C ** 2,
        bias_lr_scale=C,
        bias_adam_eps=1e-8 * D_EMB * C,
    )


def run_id(config: SAERunConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


BASELINE = SAERunConfig()
SCANS = {
    "random_state": [0, 1, 2, 3, 4],
    "l1_coefficient": [round(L1_BASELINE * m, 4) for m in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)],
    "n_slides": [1, 2, 5, 10, 20, 40],
    "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
    "expansion_factor": [8, 16, 32, 64, 128],
    "batch_size": [512, 1024, 2048, 4096, 8192],
}
JOINT_SCANS = {
    # Smaller dictionaries with lambda adjusted
    ("expansion_factor", "l1_coefficient"): [
        (e, round(L1_BASELINE * m, 4)) for e in (16, 32) for m in (0.5, 0.75, 1.5)
    ],
    # Steps needed vs dataset size (40 slides is the num_steps scan)
    ("n_slides", "num_steps"): [
        (n, s) for n in (1, 5) for s in (500, 1000, 4000, 8000, 16000)
    ],
    # Seed reproducibility after convergence
    ("random_state", "num_steps"): [(r, 16000) for r in (1, 2)],
    # Seed reproducibility after convergence at the paper's L0 (lambda from train_SAE_matched_l0.py)
    ("random_state", "num_steps", "l1_coefficient"): [(r, 16000, 7.9027) for r in (1, 2)],
}


def all_configs() -> list[SAERunConfig]:
    """Scan configs in priority order, deduplicated."""
    configs = [BASELINE]
    for field, values in SCANS.items():
        configs += [replace(BASELINE, **{field: v}) for v in values]
    for names, points in JOINT_SCANS.items():
        configs += [replace(BASELINE, **dict(zip(names, p))) for p in points]
    return list(dict.fromkeys(configs))


def scan_membership(config: SAERunConfig) -> str:
    """Scans a config belongs to, e.g. 'baseline;random_state;l1_coefficient'."""
    diff = {f.name for f in fields(SAERunConfig)
            if getattr(config, f.name) != getattr(BASELINE, f.name)}
    if not diff:
        return "baseline;" + ";".join(SCANS)
    if len(diff) == 1:
        return diff.pop()
    return "joint:" + "+".join(sorted(diff))


# ── evaluation settings ──────────────────────────────────────────────────────
PCT_THRESHOLD = 1e-3                 # fraction of patches active (notebook FeatureSelector)
MAX_SCORE_THRESHOLD = 0.01           # at input_norm='sqrt_d' scale; multiplied by C for 'd'
PCT_THRESHOLD_ALTERNATIVES = (3e-4, 3e-3)
MATCH_COS = 0.9                      # decoder cosine for a close match
N_PATCH_SUBSAMPLE = 100_000          # patches for activation correlation
EVAL_CHUNK = 8192
REQUIRED_METRIC = "fvu_heldout_slides"   # runs whose metrics.json lacks this are re-evaluated
PAPER_FEATURES = {                   # features cited in the manuscript
    "F37298_small_BV": 37298, "F7815_resp_epithelium": 7815, "F59985_LA": 59985,
    "F47380_intraalveolar_macs": 47380, "F5196_MNGC": 5196, "F56436_MNGC": 56436,
    "F46327_MNGC": 46327, "F61987_MNGC": 61987,
}


def load_sae(path):
    """joblib.load with any CUDA tensors mapped to CPU, so models saved on any GPU load anywhere."""
    torch_load = torch.load
    torch.load = functools.partial(torch_load, map_location="cpu")
    try:
        return joblib.load(path)
    finally:
        torch.load = torch_load


def training_slide_indices(n_slides: int, n_total: int) -> list[int]:
    """Evenly spaced slide positions, e.g. 1 -> [20], 2 -> [10, 30] for 40 slides."""
    return sorted({int((i + 0.5) * n_total / n_slides) for i in range(n_slides)})


@torch.no_grad()
def dataset_stats(sae, per_slide, device):
    """Per-feature activation frequency and max score, mean L0, and per-slide
    reconstruction sums from which FVU over any subset of slides is computed."""
    model = sae.model_.to(device).eval()
    s = float(sae.scale_factor_)
    n_active = torch.zeros(model.hidden_dim, device=device, dtype=torch.float64)
    max_score = torch.zeros(model.hidden_dim, device=device)
    l0 = 0.0
    n_slides, d = len(per_slide), per_slide[0].shape[1]
    sse, sum_sq, n = np.zeros(n_slides), np.zeros(n_slides), np.zeros(n_slides)
    sum_x = np.zeros((n_slides, d))
    for k, Xs in enumerate(per_slide):
        sum_x_k = torch.zeros(d, device=device, dtype=torch.float64)
        for i in range(0, len(Xs), EVAL_CHUNK):
            x = torch.from_numpy(Xs[i:i + EVAL_CHUNK]).to(device) * s
            x_hat, h = model(x)
            active = h > 0
            n_active += active.sum(0)
            max_score = torch.maximum(max_score, h.max(0).values)
            l0 += active.sum().item()
            sse[k] += ((x_hat - x) ** 2).sum().item()
            sum_sq[k] += (x.double() ** 2).sum().item()
            sum_x_k += x.sum(0).double()
        sum_x[k], n[k] = sum_x_k.cpu().numpy(), len(Xs)
    model.cpu()
    return dict(
        pct_active=(n_active / n.sum()).cpu().numpy(),
        max_score=max_score.cpu().numpy(),
        mean_l0=l0 / n.sum(),
        per_slide=dict(sse=sse, sum_sq=sum_sq, sum_x=sum_x, n=n),
    )


def fvu_over(per_slide_sums, idx):
    """FVU over the given slides: residual sum of squares / total sum of squares."""
    if len(idx) == 0:
        return float("nan")
    p = {k: v[idx] for k, v in per_slide_sums.items()}
    total = p["sum_sq"].sum() - (p["sum_x"].sum(0) ** 2).sum() / p["n"].sum()
    return float(p["sse"].sum() / total)


@torch.no_grad()
def encode_columns(sae, X, cols, device):
    """Activations of the given features on X, shape (len(X), len(cols))."""
    model = sae.model_.to(device).eval()
    s = float(sae.scale_factor_)
    cols_t = torch.as_tensor(np.asarray(cols), device=device)
    out = [model.encode(torch.from_numpy(X[i:i + EVAL_CHUNK]).to(device) * s)[:, cols_t].cpu()
           for i in range(0, len(X), EVAL_CHUNK)]
    model.cpu()
    return torch.cat(out).numpy()


def column_pearson(a, b):
    """Pearson r between matching columns; nan where a column is constant."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(0) / denom, np.nan)


def unit_decoder(sae) -> torch.Tensor:
    """Decoder columns normalized to unit length, shape (d, n_features)."""
    return F.normalize(sae.model_.decoder.weight.detach().float().cpu(), dim=0)


def plot_selection(stats, selected, pct_thr, max_thr, title, path):
    pct, mx = stats["pct_active"], stats["max_score"]
    alive = (pct > 0) & (mx > 0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pct[alive & ~selected], mx[alive & ~selected], s=3, c="tab:blue", rasterized=True)
    ax.scatter(pct[selected], mx[selected], s=3, c="tab:orange", rasterized=True)
    ax.axvline(pct_thr, ls="--", c="gray")
    ax.axhline(max_thr, ls="--", c="gray")
    ax.set(xscale="log", yscale="log", xlabel="Fraction of patches active",
           ylabel="Max feature score", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


class Evaluator:
    """Evaluates a trained SAE on all slides against the paper model (ref_SAE)."""

    def __init__(self, X, per_slide, device):
        self.per_slide = per_slide
        self.device = device
        ref = load_sae(REF_MODEL)
        paper = pd.read_csv(REF_FEATURES, index_col=0)
        self.ref_idx = paper["feature_idx"].to_numpy(int)
        self.ref_group = paper["revised_cluster"].to_numpy(int)
        self.W_ref = unit_decoder(ref)[:, self.ref_idx]
        rng = np.random.default_rng(0)
        self.sub = np.sort(rng.choice(len(X), size=min(N_PATCH_SUBSAMPLE, len(X)), replace=False))
        self.X_sub = X[self.sub]
        self.ref_acts = encode_columns(ref, self.X_sub, self.ref_idx, device)

    def __call__(self, sae, train_idx, title, run_dir):
        act_scale = C if sae.input_norm == "d" else 1.0
        max_thr = MAX_SCORE_THRESHOLD * act_scale
        stats = dataset_stats(sae, self.per_slide, self.device)
        all_idx = np.arange(len(self.per_slide))
        heldout_idx = np.setdiff1d(all_idx, train_idx)
        pct, mx = stats["pct_active"], stats["max_score"]
        selected = (pct > PCT_THRESHOLD) & (mx > max_thr)
        sel = np.where(selected)[0]
        np.savez_compressed(run_dir / "stats.npz", pct_active=pct, max_score=mx)
        plot_selection(stats, selected, PCT_THRESHOLD, max_thr, title, run_dir / "feature_selection.png")

        m = dict(
            fvu=fvu_over(stats["per_slide"], all_idx),
            fvu_train_slides=fvu_over(stats["per_slide"], np.asarray(train_idx)),
            fvu_heldout_slides=fvu_over(stats["per_slide"], heldout_idx),
            mean_l0=stats["mean_l0"],
            n_features=int(len(pct)),
            n_alive=int((pct > 0).sum()),
            n_selected=int(len(sel)),
            pct_threshold=PCT_THRESHOLD,
            max_score_threshold=max_thr,
            **{f"n_selected_pct{t:g}": int(((pct > t) & (mx > max_thr)).sum())
               for t in PCT_THRESHOLD_ALTERNATIVES},
        )

        W_run = unit_decoder(sae)
        best_any = (self.W_ref.T @ W_run).max(1).values.numpy()       # paper -> whole dictionary
        m["ref_recall_any"] = float((best_any > MATCH_COS).mean())
        if len(sel) == 0:
            m.update(median_best_cos_to_ref=np.nan, n_selected_matched=0,
                     ref_recall=0.0, ref_median_best_cos=np.nan, median_patch_r=np.nan)
            return m

        cos = (W_run[:, sel].T @ self.W_ref).numpy()                   # (n_sel, 361)
        run_best = cos.max(1)                                         # run -> paper
        ref_best = cos.max(0)                                         # paper -> run selected
        ref_match = sel[cos.argmax(0)]
        m.update(
            median_best_cos_to_ref=float(np.median(run_best)),
            n_selected_matched=int((run_best > MATCH_COS).sum()),
            frac_selected_matched=float((run_best > MATCH_COS).mean()),
            ref_recall=float((ref_best > MATCH_COS).mean()),
            ref_median_best_cos=float(np.median(ref_best)),
        )
        # Patch-level agreement of each paper feature with its best-matching feature
        run_acts = encode_columns(sae, self.X_sub, ref_match, self.device)
        r = column_pearson(self.ref_acts, run_acts)
        m["median_patch_r"] = float(np.nanmedian(r))
        m["frac_patch_r_gt_0p8"] = float(np.nanmean(r > 0.8))
        groups = np.unique(self.ref_group)
        m["n_groups_recovered"] = int(sum((ref_best[self.ref_group == g] > MATCH_COS).any() for g in groups))
        for g in groups:
            m[f"group{g}_recall"] = float((ref_best[self.ref_group == g] > MATCH_COS).mean())
        pos = {f: i for i, f in enumerate(self.ref_idx)}
        for name, f in PAPER_FEATURES.items():
            m[f"cos_{name}"] = float(ref_best[pos[f]])
            m[f"patch_r_{name}"] = float(r[pos[f]])
        return m


def plot_scan_summary(df, path):
    """One column per scanned hyperparameter; shaded band = range over seeds at baseline."""
    metrics = ["n_selected", "fvu", "mean_l0", "median_best_cos_to_ref",
               "frac_selected_matched", "ref_recall", "median_patch_r"]
    scan_fields = list(SCANS)
    in_scan = lambda field: df["scan"].str.split(";").apply(lambda scans: field in scans)
    seeds = df[in_scan("random_state")]
    fig, axes = plt.subplots(len(metrics), len(scan_fields),
                             figsize=(3 * len(scan_fields), 2.2 * len(metrics)), squeeze=False)
    for j, field in enumerate(scan_fields):
        sub = df[in_scan(field)].sort_values(field)
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            if len(seeds) and field != "random_state":
                ax.axhspan(seeds[metric].min(), seeds[metric].max(), color="0.85")
            ax.plot(sub[field], sub[metric], "o-")
            ax.axvline(getattr(BASELINE, field), ls=":", c="gray")
            if field in ("l1_coefficient", "learning_rate", "num_steps", "batch_size",
                         "expansion_factor", "n_slides"):
                ax.set_xscale("log")
            if i == len(metrics) - 1:
                ax.set_xlabel(field)
            if j == 0:
                ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def seed_pairwise(df, path):
    """Pairwise matching of selected features between runs that differ only in random_state."""
    other = [f.name for f in fields(SAERunConfig) if f.name != "random_state"]
    rows = []
    for key, group in df.groupby(other):
        if len(group) < 2:
            continue
        W = {}
        for _, row in group.iterrows():
            run_dir = RUNS_DIR / row["run_id"]
            stats = np.load(run_dir / "stats.npz")
            sel = np.where((stats["pct_active"] > PCT_THRESHOLD) & (stats["max_score"] > row["max_score_threshold"]))[0]
            W[row["random_state"]] = unit_decoder(load_sae(run_dir / "sae.joblib"))[:, sel]
        seeds = sorted(W)
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                best = (W[a].T @ W[b]).max(1).values.numpy()   # a's features -> b's features
                rows.append({**dict(zip(other, key)), "seed_a": a, "seed_b": b,
                             "n_selected_a": W[a].shape[1], "n_selected_b": W[b].shape[1],
                             "frac_matched": float((best > MATCH_COS).mean()),
                             "median_best_cos": float(np.median(best))})
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_steps_by_slides(df, path):
    """Metrics vs num_steps, one line per number of training slides."""
    metrics = ["fvu", "fvu_heldout_slides", "mean_l0", "n_selected", "ref_recall",
               "median_best_cos_to_ref", "median_patch_r"]
    other = [f.name for f in fields(SAERunConfig) if f.name not in ("n_slides", "num_steps")]
    base = df[(df[other] == pd.Series({k: getattr(BASELINE, k) for k in other})).all(axis=1)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3), squeeze=False)
    for n_slides, sub in base.groupby("n_slides"):
        sub = sub.sort_values("num_steps")
        if len(sub) < 3:
            continue
        for ax, metric in zip(axes[0], metrics):
            ax.plot(sub["num_steps"], sub[metric], "o-", label=f"{n_slides} slides")
    for ax, metric in zip(axes[0], metrics):
        ax.set(xscale="log", xlabel="num_steps", title=metric)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def load_data():
    """Embeddings of all slides as one array, per-slide views into it, and slide ids."""
    slides = ezslide.open_slides(load_manifest(), attach_images=False)
    per_slide = [t.obsm["UNI_embedding"] for _, t in SlideSource(slides, tile_key="tiles")]
    X = np.vstack(per_slide).astype(np.float32, copy=False)
    per_slide = np.split(X, np.cumsum([len(a) for a in per_slide])[:-1])  # views into X
    print(f"{len(per_slide)} slides, {X.shape} patches", flush=True)
    return X, per_slide, list(slides.keys())


def metrics_current(run_dir):
    path = run_dir / "metrics.json"
    return path.exists() and REQUIRED_METRIC in json.loads(path.read_text())


def run_config(config, data, evaluator, device, eval_only=False, max_train_slides=None):
    """Train (if needed) and evaluate one config; returns its metrics, or None if skipped."""
    X, per_slide, slide_ids = data
    rid = run_id(config)
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    idx = training_slide_indices(config.n_slides, len(per_slide))
    trained = (run_dir / "sae.joblib").exists()
    if trained and metrics_current(run_dir):
        return json.loads((run_dir / "metrics.json").read_text())
    if not trained and (eval_only or (max_train_slides is not None and config.n_slides > max_train_slides)):
        return None
    # Lock so that concurrent scan processes do not work on the same run
    lock = run_dir / ".lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        print(f"skip (locked): {rid}", flush=True)
        return None
    try:
        if trained:
            sae = load_sae(run_dir / "sae.joblib")
        else:
            print(f"training {rid} [{scan_membership(config)}] {asdict(config)}", flush=True)
            X_train = X if len(idx) == len(per_slide) else np.vstack([per_slide[i] for i in idx])
            t0 = time.time()
            sae = SparseAutoencoder(**sae_kwargs(config)).fit(X_train, device=device, verbose=False)
            train_minutes = (time.time() - t0) / 60
            sae.model_.cpu()
            joblib.dump(sae, run_dir / "sae.joblib")
            (run_dir / "train_info.json").write_text(json.dumps(dict(
                train_minutes=train_minutes,
                n_train_patches=int(sum(len(per_slide[i]) for i in idx)),
                training_slides=[slide_ids[i] for i in idx],
                final_recon_loss=float(np.mean(sae._training_log["recon_losses"][-100:])),
                final_sparsity_loss=float(np.mean(sae._training_log["sparsity_losses"][-100:])),
            ), indent=2))
        metrics = evaluator(sae, idx, f"{rid} {scan_membership(config)}", run_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  {rid} n_selected {metrics['n_selected']}, fvu {metrics['fvu']:.4f}, "
              f"mean_l0 {metrics['mean_l0']:.2f}, ref_recall {metrics['ref_recall']:.3f}, "
              f"median_best_cos_to_ref {metrics['median_best_cos_to_ref']:.3f}", flush=True)
        del sae
        torch.cuda.empty_cache()
        return metrics
    finally:
        lock.unlink()


def write_summary():
    """summary.csv and figures over every evaluated run in RUNS_DIR."""
    rows = []
    for metrics_path in sorted(RUNS_DIR.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        config = SAERunConfig(**json.loads((run_dir / "config.json").read_text()))
        info = run_dir / "train_info.json"
        rows.append({"run_id": run_dir.name, "scan": scan_membership(config), **asdict(config),
                     **(json.loads(info.read_text()) if info.exists() else {}),
                     **json.loads(metrics_path.read_text())})
    df = pd.DataFrame(rows).drop(columns=["training_slides"], errors="ignore")
    df.to_csv(RUNS_DIR / "summary.csv", index=False)
    plot_scan_summary(df, RUNS_DIR / "scan_summary.png")
    plot_steps_by_slides(df, RUNS_DIR / "steps_by_slides.png")
    seed_pairwise(df, RUNS_DIR / "seed_pairwise.csv")
    print(f"wrote {RUNS_DIR / 'summary.csv'} ({len(df)} runs)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dry-run", action="store_true", help="list configs and exit")
    parser.add_argument("--eval-only", action="store_true", help="evaluate existing runs only")
    parser.add_argument("--max-train-slides", type=int, default=None,
                        help="only train configs with at most this many slides (limits memory)")
    args = parser.parse_args()

    configs = all_configs()
    print(f"{len(configs)} configs (baseline lambda {L1_BASELINE})")
    if args.dry_run:
        for c in configs:
            print(run_id(c), scan_membership(c), asdict(c))
        return

    data = load_data()
    evaluator = Evaluator(data[0], data[1], args.device)
    # New runs first, then re-evaluation of runs with outdated metrics
    todo = sorted(configs, key=lambda c: (RUNS_DIR / run_id(c) / "sae.joblib").exists())
    for config in todo:
        run_config(config, data, evaluator, args.device, args.eval_only, args.max_train_slides)
    write_summary()


if __name__ == "__main__":
    main()
