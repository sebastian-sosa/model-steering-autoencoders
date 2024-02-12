from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from models.toy_model import ToyModel
from torch import Tensor, nn


def linear_lr(step, steps):
    return (1 - (step / steps))


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False
    weight_normalize_eps: float = 1e-8


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig):
        """
        Initializes the two weights and biases according to the type signature above.

        If self.cfg.tied = True, then we only create W_enc, not W_dec.
        """
        super().__init__()
        self.cfg = cfg

        self.W_enc = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))
            )
        )
        if not (cfg.tied_weights):
            self.W_dec = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))
                )
            )

        self.b_enc = nn.Parameter(torch.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.n_instances, cfg.n_input_ae))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def normalize_and_return_W_dec(
        self,
    ) -> Float[Tensor, "n_instances n_hidden_ae n_input_ae"]:
        """
        If self.cfg.tied = True, we return the normalized & transposed encoder weights.
        If self.cfg.tied = False, we normalize the decoder weights in-place, and return them.

        Normalization should be over the `n_input_ae` dimension, i.e. each feature should have a noramlized decoder weight.
        """
        if self.cfg.tied_weights:
            return self.W_enc.transpose(-1, -2) / (
                self.W_enc.transpose(-1, -2).norm(dim=1, keepdim=True)
                + self.cfg.weight_normalize_eps
            )
        else:
            self.W_dec.data = self.W_dec.data / (
                self.W_dec.data.norm(dim=2, keepdim=True)
                + self.cfg.weight_normalize_eps
            )
            return self.W_dec

    def forward(self, h: Float[Tensor, "batch_size n_instances n_input_ae"]):
        """
        Runs a forward pass on the autoencoder, and returns several outputs.

        Inputs:
            h: Float[Tensor, "batch_size n_instances n_hidden"]
                hidden activations generated from a Model instance

        Returns:
            l1_loss: Float[Tensor, "batch_size n_instances"]
                L1 loss for each batch elem & each instance (sum over the `n_hidden_ae` dimension)
            l2_loss: Float[Tensor, "batch_size n_instances"]
                L2 loss for each batch elem & each instance (take mean over the `n_input_ae` dimension)
            loss: Float[Tensor, ""]
                Sum of L1 and L2 loss (with the former scaled by `self.cfg.l1_coeff1). We sum over the `n_instances`
                dimension but take mean over the batch dimensio
            acts: Float[Tensor, "batch_size n_instances n_hidden_ae"]
                Activations of the autoencoder's hidden states (post-ReLU)
            h_reconstructed: Float[Tensor, "batch_size n_instances n_input_ae"]
                Reconstructed hidden states, i.e. the autoencoder's final output
        """
        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent,
            self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae",
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = (
            einops.einsum(
                acts,
                self.normalize_and_return_W_dec(),
                "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae",
            )
            + self.b_dec
        )

        # Compute loss, return values
        l2_loss = (
            (h_reconstructed - h).pow(2).mean(-1)
        )  # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1)  # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum()  # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    def optimize(
        self,
        model: ToyModel,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        """
        Optimizes the autoencoder using the given hyperparameters.

        This function should take a trained model as input.
        """
        if neuron_resample_window is not None:
            assert (dead_neuron_window is not None) and (
                dead_neuron_window < neuron_resample_window
            )

        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {
            "W_enc": [],
            "W_dec": [],
            "colors": [],
            "titles": [],
            "frac_active": [],
        }
        colors = None
        title = "no resampling yet"

        for step in progress_bar:

            # Update learning rate based on `lr_scale` function
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Get a batch of hidden activations from the model (for the training step & the neuron resampling)
            with torch.inference_mode():
                features = model.generate_batch(batch_size)
                h = einops.einsum(
                    features,
                    model.W,
                    "batch instances feats, instances hidden feats -> batch instances hidden",
                )

            # Resample dead neurons
            if (neuron_resample_window is not None) and (
                (step + 1) % neuron_resample_window == 0
            ):
                # Get the fraction of neurons active in the previous window
                frac_active_in_window = torch.stack(
                    frac_active_list[-neuron_resample_window:], dim=0
                )
                # Apply resampling
                colors, title = self.resample_neurons(
                    h, frac_active_in_window, neuron_resample_scale
                )

            # Optimize
            l1_loss, l2_loss, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Calculate the mean sparsities over batch dim for each (instance, feature)
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    l1_loss=self.cfg.l1_coeff * l1_loss.mean(0).sum().item(),
                    l2_loss=l2_loss.mean(0).sum().item(),
                    lr=step_lr,
                )
                data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                data_log["W_dec"].append(
                    self.normalize_and_return_W_dec().detach().cpu().clone()
                )
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}: {title}")
                data_log["frac_active"].append(frac_active.detach().cpu().clone())

        return data_log

    @torch.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size n_instances n_hidden"],
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> Tuple[List[List[str]], str]:
        """
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        """
        # Create an object to store the dead neurons (this will be useful for plotting)
        dead_features_mask = torch.empty(
            (self.cfg.n_instances, self.cfg.n_hidden_ae),
            dtype=torch.bool,
            device=self.W_enc.device,
        )

        for instance in range(self.cfg.n_instances):

            # Find the dead neurons in this instance. If all neurons are alive, continue
            is_dead = frac_active_in_window[:, instance].sum(0) < 1e-8
            dead_features_mask[instance] = is_dead
            dead_features = torch.nonzero(is_dead).squeeze(-1)
            n_dead = dead_features.numel()
            if n_dead == 0:
                continue

            # Get our randomized replacement values
            replacement_values = torch.randn(
                (n_dead, self.cfg.n_input_ae), device=self.W_enc.device
            )  # shape [n_dead n_input_ae]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True) + 1e-8
            )

            # Lastly, set the new weights & biases
            self.W_dec.data[instance, dead_features, :] = replacement_values_normalized
            self.W_enc.data[instance, :, dead_features] = (
                replacement_values_normalized.T
            )
            self.b_enc.data[instance, dead_features] = 0.0

        # Return data for visualising the resampling process
        colors = [
            ["red" if dead else "black" for dead in dead_neuron_mask_inst]
            for dead_neuron_mask_inst in dead_features_mask
        ]
        title = f"resampling {dead_features_mask.sum()}/{dead_features_mask.numel()} neurons (shown in red)"
        return colors, title
