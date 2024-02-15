# %%
import os
from typing import Tuple

from huggingface_hub import snapshot_download
from sae_training.activations_store import ActivationsStore
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import LMSparseAutoencoderSessionloader
from transformer_lens import HookedTransformer


def load_models(
    repo_id: str = "jbloom/GPT2-Small-SAEs",
    autoencoder_layer: int = 8,
) -> Tuple[HookedTransformer, SparseAutoencoder, ActivationsStore]:
    repo_dir = snapshot_download(repo_id)
    model_files = [f for f in os.listdir(repo_dir) if f.endswith('.pt')]
    model_weights = []
    feature_sparsity = []

    for model_file in model_files:
        if 'log_feature_sparsity' in model_file:
            feature_sparsity.append(model_file)
        else:
            model_weights.append(model_file)

    weights_path = os.path.join(repo_dir, model_weights[autoencoder_layer])
    model, sparse_autoencoder, activations_loader = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        weights_path
    )
    return model, sparse_autoencoder, activations_loader

model, sparse_autoencoder, activations_loader = load_models()
