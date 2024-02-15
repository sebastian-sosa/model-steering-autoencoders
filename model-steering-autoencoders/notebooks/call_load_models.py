# %%
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(f'{parent_dir}/models')
# %%
from models.load_models import load_models

# %%
model, sparse_autoencoder, activations_loader = load_models()
