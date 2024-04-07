"""Tests for the VAE and data pipeline."""

import numpy as np
import pytest
import torch

from src.data.dataset import SensorWindowDataset
from src.data.preprocess import (
    add_rul_labels,
    create_windows,
    normalize_per_engine,
)
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE, vae_loss


@pytest.fixture
def sample_df():
    """Fake engine data for testing."""
    import pandas as pd

    rows = []
    for eid in [1, 2]:
        n_cycles = 50
        for c in range(1, n_cycles + 1):
            row = {"engine_id": eid, "cycle": c}
            for i in range(1, 15):
                row[f"sensor_{i}"] = np.random.rand() * 100
            rows.append(row)
    return pd.DataFrame(rows)


class TestVAE:
    def test_forward_shape(self):
        model = VAE(input_dim=420, hidden_dims=[64, 32], latent_dim=16)
        x = torch.randn(8, 420)
        recon, mu, logvar = model(x)

        assert recon.shape == (8, 420)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)

    def test_loss_positive(self):
        model = VAE(input_dim=100, hidden_dims=[32], latent_dim=8)
        x = torch.rand(4, 100)
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar)

        assert loss.item() > 0
        assert recon_l.item() >= 0
        assert kl_l.item() >= 0

    def test_latent_deterministic(self):
        model = VAE(input_dim=100, hidden_dims=[32], latent_dim=8)
        model.eval()
        x = torch.rand(4, 100)
        z1 = model.get_latent(x)
        z2 = model.get_latent(x)
        assert torch.allclose(z1, z2)


class TestAutoencoder:
    def test_forward_shape(self):
        model = Autoencoder(input_dim=420, hidden_dims=[64, 32], latent_dim=16)
        x = torch.randn(8, 420)
        out = model(x)
        assert out.shape == (8, 420)


class TestPreprocessing:
    def test_add_rul_labels(self, sample_df):
        df = add_rul_labels(sample_df, rul_cap=125)
        assert "rul" in df.columns
        # last cycle of each engine should have RUL = 0
        for eid in df.engine_id.unique():
            engine = df[df.engine_id == eid]
            last = engine[engine.cycle == engine.cycle.max()]
            assert last.rul.values[0] == 0

    def test_normalize_per_engine(self, sample_df):
        sensor_cols = [c for c in sample_df.columns if c.startswith("sensor_")]
        normed = normalize_per_engine(sample_df)
        for eid in normed.engine_id.unique():
            engine = normed[normed.engine_id == eid]
            vals = engine[sensor_cols].values
            # each sensor should be roughly in [0, 1]
            assert vals.min() >= -0.01
            assert vals.max() <= 1.01

    def test_create_windows(self, sample_df):
        df = add_rul_labels(sample_df)
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        windows, ruls, eids = create_windows(
            df, window_size=10, stride=5, sensor_cols=sensor_cols
        )

        assert windows.ndim == 3
        assert windows.shape[1] == 10
        assert windows.shape[2] == len(sensor_cols)
        assert len(ruls) == len(windows)


class TestDataset:
    def test_dataset_len(self):
        windows = np.random.rand(100, 30, 14).astype(np.float32)
        ds = SensorWindowDataset(windows)
        assert len(ds) == 100

    # TODO: add test for RUL labels in dataset
