"""Convert the legacy per-SAE-feature token clusterizers to the current mesoslide API.

Source: .../SAE_paper/TB/3D_HnE/relu_sae/token_classifier/
    feature_<id>.joblib   -- a bare fitted sklearn.cluster.KMeans (pickled on
                              its own, not wrapped in the old
                              meson.tools.segmenters.TokenClusterizer class --
                              that old class pickles fine under the current
                              sklearn/numpy versions, unlike the ref_SAE.joblib
                              case convert_ref_sae.py deals with).
    positive_label_map.npz -- {feature_id (int) -> cluster_order (np.ndarray)},
                              the differential-abundance ordering the old
                              TokenClusterizer.fit() computed per SAE feature.
Together, one KMeans + its cluster_order is everything
`mesoslide.tools.segmenters.TokenClusterizer` needs besides the (shared)
vision model -- all 199 feature ids present in positive_label_map.npz were
clustered against the same UNI embedding (matching this pipeline's
UNI_SAE_<id> feature-naming convention elsewhere, and the old class's
meson.tools.embedders.UNI.UNIEmbedder).

Each output file is a real `TokenClusterizer` instance, saved directly with
joblib.dump -- current `TokenClusterizer` doesn't hold onto its vision model
(only reads `grid_size`/`patch_size` off it at construction time; see its
class docstring), so this stays cheap (~33 KB/file) despite going through
`model=` at construction. Resolving the (shared) UNI model still costs
something once per process -- there's no way around needing it to read
grid_size/patch_size -- but it's resolved exactly once here and reused for
every one of the ~200 constructions, not reloaded per feature.

Usage:
    export HF_TOKEN=hf_...
    uv run python convert_old_clusterizers.py
"""

from fmsae_repro import MODELS, LEGACY_ANALYSIS
import os
import re
from glob import glob

import joblib
import numpy as np

SRC_DIR = (
    f"{LEGACY_ANALYSIS}/SAE_paper/TB/3D_HnE/"
    "relu_sae/token_classifier"
)
DST_DIR = f"{MODELS}/token_clusterers/legacy"
MODEL_NAME = "uni"


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    from lazyslide_models import MODEL_REGISTRY
    from mesoslide.tools.segmenters import TokenClusterizer

    # Resolved once, reused (not stored) across every TokenClusterizer built below.
    model = MODEL_REGISTRY[MODEL_NAME](token=os.environ.get("HF_TOKEN"))

    cluster_orders = np.load(
        os.path.join(SRC_DIR, "positive_label_map.npz"), allow_pickle=True,
    )["positive_label_map"].item()

    feature_paths = sorted(glob(os.path.join(SRC_DIR, "feature_*.joblib")))
    print(f"Found {len(feature_paths)} legacy clusterizers, "
          f"{len(cluster_orders)} entries in positive_label_map.npz")

    converted, skipped = [], []
    for path in feature_paths:
        m = re.match(r"feature_(\d+)\.joblib$", os.path.basename(path))
        feature_id = int(m.group(1))

        if feature_id not in cluster_orders:
            skipped.append(feature_id)
            continue

        feature_name = f"UNI_SAE_{feature_id}"
        clusterizer = TokenClusterizer(
            model=model,
            kmeans=joblib.load(path),
            cluster_order=cluster_orders[feature_id],
            feature_name=feature_name,
            device="cpu",
        )

        out_path = os.path.join(DST_DIR, f"{feature_name}.joblib")
        joblib.dump(clusterizer, out_path)
        converted.append(feature_name)

    print(f"Converted {len(converted)} clusterizers to {DST_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} feature ids with no cluster_order entry: {skipped}")

    # Sanity check: reload one converted clusterizer from disk and run
    # transform() on a synthetic batch through the real model, end to end.
    import torch
    from mesoslide.tools._model_stage import ImageModelStage

    check_path = os.path.join(DST_DIR, f"{converted[0]}.joblib")
    clusterizer = joblib.load(check_path)

    # model_name must round-trip back into MODEL_REGISTRY (it's what
    # extract_cluster_maps/fit() use to auto-resolve a model when the caller
    # doesn't pass one) -- not just equal MODEL_NAME by construction, but
    # actually be a valid registry key.
    assert clusterizer.model_name == MODEL_NAME
    assert clusterizer.model_name in MODEL_REGISTRY

    images = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8)
    fm_stage = ImageModelStage(model, dense=True, device="cpu")
    masks = clusterizer.transform(fm_stage(images), output_size=(224, 224))
    assert masks.shape == (2, 224, 224) and masks.dtype == np.uint8
    print(f"Sanity check passed on '{converted[0]}': transform() -> {masks.shape} {masks.dtype}, "
          f"model_name={clusterizer.model_name!r}")


if __name__ == "__main__":
    main()
