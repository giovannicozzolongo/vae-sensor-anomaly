"""Variational Autoencoder for sensor anomaly detection."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """VAE with MLP encoder/decoder.

    Args:
        input_dim: flattened window size (window_size * n_features)
        hidden_dims: list of hidden layer sizes
        latent_dim: dimension of latent space
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder
        enc_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # decoder
        dec_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        # sigmoid since inputs are min-max normalized to [0,1]
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean) without sampling."""
        mu, _ = self.encode(x)
        return mu


def vae_loss(
    recon_x,
    x,
    mu,
    logvar,
    kl_weight: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO loss = reconstruction + KL divergence.

    free_bits: minimum KL per latent dimension to prevent collapse.
    set to 0 to disable.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")

    # KL per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.mean()

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss
