# Reproducing Zhao et al. (2026): foundation-model sparse autoencoders for histology

Scripts and notebooks that reproduce the analyses in *Compositional and interpretable
representation of histology using AI foundation models and sparse autoencoders*
([bioRxiv](https://doi.org/10.64898/2026.06.03.725182)), from raw H&E whole-slide images to
the paper's figures and tables. The analysis code itself lives in the
[`mesoslide`](https://github.com/labsyspharm/mesoslide) package; this repository only calls it.
For a minimal example of applying the pretrained SAE to new slides, see
[Zhao-FMSAE-2026](https://github.com/labsyspharm/Zhao-FMSAE-2026).

## Layout

```
src/fmsae_repro/   paths (repository-relative) and shared helpers
scripts/           batch steps (Slurm-ready): preprocessing, features, training, evaluation
notebooks/         training, SAE feature analysis, token-level analysis, multimodal (CyCIF)
data/              generated: slide zarr stores + manifest.csv          (not tracked)
models/            generated: SAE, LLC, dictionary-learning models,    (not tracked)
                   token clusterers
results/           generated: tables, figures, patch collections       (not tracked)
logs/              Slurm logs                                           (not tracked)
```

No file contains an absolute path. Every location is built from the repository root in
`src/fmsae_repro/paths.py`:

```python
from fmsae_repro import DATA, MODELS, RESULTS, load_manifest

slides = ezslide.open_slides(load_manifest(), attach_images=False)   # data/manifest.csv
sae = joblib.load(MODELS / "ref_SAE.joblib")
table.to_csv(RESULTS / "tables" / "my_table.csv")
```

## Setup

```bash
git clone <this repository> && cd <repository>
uv sync                      # installs mesoslide and this package (editable)
cp .env.example .env         # then set FMSAE_RAW_DATA to your raw data directory
```

`FMSAE_RAW_DATA` points to the raw WSIs and CyCIF data
([Zenodo](https://zenodo.org/records/21299341)); it is expected to contain `3D_HnE/`,
`3D_adjacent_registered_HnE/` and `3D/` (CyCIF images, segmentation, quantification,
`markers.csv`). If unset, `raw_data/` inside the repository is used.

Slurm scripts are submitted from the repository root, e.g.
`sbatch scripts/preprocessing/preprocess_wsi.sh`.

## Pipeline

| Step | Code | Produces |
|---|---|---|
| 1. Tile and embed WSIs (UNI) | `scripts/preprocessing/preprocess_wsi.py` (+ `preprocess_registered_wsi.py` for the registered adjacent section) | `data/<slide>.zarr`, `data/adjacent_registered/` |
| 2. Manifest | `notebooks/sae_features/visualize_sae.ipynb` (manifest cell) | `data/manifest.csv` |
| 3. SAE feature scores | `scripts/features/extract_sae_features.py` | SAE scores in each zarr's tile table |
| 4. Token label maps | `scripts/features/extract_token_labels.py` | `results/token_maps/` |
| 5. Train models | `notebooks/training/train_SAE.ipynb`, `train_LLC.ipynb`, `train_DL.ipynb`; sweep: `scripts/training/train_SAE_scan.py` | `models/`, `results/tables/DL_resource_usage.csv` |
| 6. Reconstruction (FVU) | `scripts/evaluation/recon_fvu.py` | printed FVU / L0 for SAE1 and the DL model |
| 7. SAE feature analysis | `notebooks/sae_features/` | feature selection, clustering, galleries |
| 8. Token-level analysis | `notebooks/token_level/` | `models/token_clusterers/`, `results/SAE_feature_high_scoring_patches/`, `results/tables/Supplementary_Table_3.csv` |
| 9. Multimodal (CyCIF) | `notebooks/multimodal/SAE_token_CyCIF_analysis.ipynb` | `results/figures/multimodal_panels/` |

`scripts/legacy_conversion/` converts models saved by earlier versions of the code
(`ref_SAE.joblib`, token clusterers) to the current `mesoslide` API; it reads those
earlier outputs from `FMSAE_LEGACY_ANALYSIS` and is not needed when training from scratch.

## Citation

Zhao, Z. et al. (2026). Compositional and interpretable representation of histology using AI
foundation models and sparse autoencoders. bioRxiv. https://doi.org/10.64898/2026.06.03.725182
