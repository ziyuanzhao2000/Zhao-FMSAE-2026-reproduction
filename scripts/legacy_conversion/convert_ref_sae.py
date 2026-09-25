"""Convert the legacy ref_SAE.joblib checkpoint to the current mesoslide API.

The old checkpoint was pickled under the pre-rename package name
`meson.tools.embedders.SAE` and its fitted `mesoslide.tools.sparse_coding.SparseAutoencoder`
/ `SimpleAutoencoder` classes have since changed (constructor args, and
`transform()` now calls `sklearn.utils.validation.validate_data` /
`check_is_fitted`, which this checkpoint's attributes don't satisfy -- see
extract_sae_features.py's `load_sae` shim for the full unpickling
compatibility story).

This script loads the checkpoint through that shim, then builds a fresh
`mesoslide.tools.sparse_coding.SparseAutoencoder` and copies the fitted state
over (inner autoencoder weights, scale factor, embedding dimension) so the
result is a real instance of the current class, usable directly with
`sae.transform(...)` under the current API. Hyperparameters that don't exist
under old names (`initial_lambda`/`final_lambda`) are mapped to their current
equivalents (`min_lambda`/`max_lambda`).

Usage:
    uv run python convert_ref_sae.py
"""

from fmsae_repro import MODELS, LEGACY_ANALYSIS
import joblib
import torch

from extract_sae_features import load_sae
from mesoslide.tools.sparse_coding import SparseAutoencoder
from mesoslide.tools.sparse_coding._sae import SimpleAutoencoder

OLD_SAE_PATH = f"{LEGACY_ANALYSIS}/TB/3D_HnE/ref_SAE.joblib"
NEW_SAE_PATH = f"{MODELS}/ref_SAE.joblib"


def convert(old):
    old_model = old.model_

    new_sae = SparseAutoencoder(
        expansion_factor=old.expansion_factor,
        batch_size=old.batch_size,
        num_steps=old.num_steps,
        min_lambda=old.initial_lambda,
        max_lambda=old.final_lambda,
        target_sparsity=old.target_sparsity,
        random_state=old.random_state,
    )

    new_model = SimpleAutoencoder(
        input_dim=old_model.input_dim, expansion_factor=old.expansion_factor
    )
    # SimpleAutoencoder._initialize_weights sets decoder.weight.data as a
    # transposed VIEW of encoder.weight's storage. Loading the checkpoint's
    # decoder.weight in place would then overwrite encoder.weight through
    # that aliased storage. Give decoder.weight its own storage first.
    new_model.decoder.weight.data = new_model.decoder.weight.data.clone()
    new_model.load_state_dict(old_model.state_dict())

    new_sae.model_ = new_model
    new_sae.scale_factor_ = old.scale_factor_
    new_sae.embed_dim_ = old.embed_dim_
    new_sae.n_features_in_ = old_model.input_dim
    return new_sae


def main():
    old = load_sae(OLD_SAE_PATH)
    new_sae = convert(old)

    # Sanity check: transform a small random batch through both models and
    # confirm they produce identical sparse activations.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, new_sae.n_features_in_)
    old_out = old.transform(x.numpy(), device=device)
    new_out = new_sae.transform(x.numpy(), device=device)
    assert (old_out != new_out).nnz == 0, "converted SAE output does not match original"

    joblib.dump(new_sae, NEW_SAE_PATH)
    print(f"Wrote converted SAE to {NEW_SAE_PATH}")


if __name__ == "__main__":
    main()
