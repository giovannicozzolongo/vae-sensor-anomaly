"""Training loops for VAE and AE models."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SensorWindowDataset
from src.data.preprocess import preprocess_subset
from src.models.autoencoder import Autoencoder
from src.models.baselines import IsolationForestDetector, OCSVMDetector
from src.models.vae import VAE, vae_loss
from src.utils.config import load_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_kl_weight(epoch: int, cfg: dict) -> float:
    """Linear KL annealing schedule."""
    anneal = cfg["vae"]["kl_annealing"]
    start_e, end_e = anneal["start_epoch"], anneal["end_epoch"]
    start_w, end_w = anneal["start_weight"], anneal["end_weight"]

    if epoch <= start_e:
        return start_w
    if epoch >= end_e:
        return end_w * cfg["vae"]["beta"]
    frac = (epoch - start_e) / (end_e - start_e)
    return (start_w + frac * (end_w - start_w)) * cfg["vae"]["beta"]


def train_vae(cfg: dict, data: dict, device: torch.device) -> VAE:
    """Train VAE on healthy windows."""
    windows = data["train_windows"]
    rul = data["train_rul"]

    # use only "healthy" windows for training (RUL > cap * 0.5)
    rul_cap = cfg["data"]["rul_cap"]
    healthy_mask = rul > rul_cap * 0.4
    train_w = windows[healthy_mask]
    logger.info(
        f"training VAE on {len(train_w)} healthy windows (out of {len(windows)})"
    )

    dataset = SensorWindowDataset(train_w, flatten=True)
    loader = DataLoader(dataset, batch_size=cfg["vae"]["batch_size"], shuffle=True)

    input_dim = train_w.shape[1] * train_w.shape[2]
    model = VAE(input_dim, cfg["vae"]["hidden_dims"], cfg["vae"]["latent_dim"]).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["vae"]["lr"])

    free_bits = cfg["vae"].get("free_bits", 0.0)
    n_epochs = cfg["vae"]["n_epochs"]
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        kl_w = get_kl_weight(epoch, cfg)

        for batch in loader:
            x = batch["input"].to(device)
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(
                recon, x, mu, logvar, kl_weight=kl_w, free_bits=free_bits
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()

        n_batches = len(loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"epoch {epoch+1}/{n_epochs} | loss: {total_loss/n_batches:.4f} | "
                f"recon: {total_recon/n_batches:.4f} | kl: {total_kl/n_batches:.4f} | kl_w: {kl_w:.3f}"
            )

    return model


def train_autoencoder(cfg: dict, data: dict, device: torch.device) -> Autoencoder:
    """Train standard AE."""
    windows = data["train_windows"]
    rul = data["train_rul"]
    rul_cap = cfg["data"]["rul_cap"]
    healthy_mask = rul > rul_cap * 0.4
    train_w = windows[healthy_mask]

    logger.info(f"training AE on {len(train_w)} healthy windows")

    dataset = SensorWindowDataset(train_w, flatten=True)
    loader = DataLoader(
        dataset, batch_size=cfg["autoencoder"]["batch_size"], shuffle=True
    )

    input_dim = train_w.shape[1] * train_w.shape[2]
    model = Autoencoder(
        input_dim, cfg["autoencoder"]["hidden_dims"], cfg["autoencoder"]["latent_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["autoencoder"]["lr"])
    criterion = torch.nn.MSELoss()

    n_epochs = cfg["autoencoder"]["n_epochs"]
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch["input"].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"AE epoch {epoch+1}/{n_epochs} | loss: {total_loss/len(loader):.4f}"
            )

    return model


def train_baselines(cfg: dict, data: dict) -> dict:
    """Train IF and OC-SVM baselines."""
    windows = data["train_windows"]
    rul = data["train_rul"]
    rul_cap = cfg["data"]["rul_cap"]
    healthy_mask = rul > rul_cap * 0.4
    train_w = windows[healthy_mask]

    results = {}

    # Isolation Forest
    if_cfg = cfg["isolation_forest"]
    iforest = IsolationForestDetector(
        n_estimators=if_cfg["n_estimators"],
        contamination=if_cfg["contamination"],
        random_state=if_cfg["random_state"],
    )
    iforest.fit(train_w)
    results["isolation_forest"] = iforest

    # OC-SVM
    ocsvm_cfg = cfg["ocsvm"]
    ocsvm = OCSVMDetector(
        kernel=ocsvm_cfg["kernel"],
        nu=ocsvm_cfg["nu"],
        gamma=ocsvm_cfg["gamma"],
    )
    ocsvm.fit(train_w)
    results["ocsvm"] = ocsvm

    return results


def get_reconstruction_scores(
    model, windows: np.ndarray, device: torch.device
) -> np.ndarray:
    """Compute per-sample reconstruction error for AE or VAE."""
    model.eval()
    dataset = SensorWindowDataset(windows, flatten=True)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            if isinstance(model, VAE):
                recon, _, _ = model(x)
            else:
                recon = model(x)
            mse = torch.mean((recon - x) ** 2, dim=1)
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores)


def get_vae_anomaly_scores(
    model: VAE, windows: np.ndarray, device: torch.device
) -> np.ndarray:
    """VAE anomaly score: reconstruction error + KL divergence per sample.

    Better than just reconstruction error because KL captures how far
    the sample is from the learned prior.
    """
    model.eval()
    dataset = SensorWindowDataset(windows, flatten=True)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            recon, mu, logvar = model(x)

            # per-sample reconstruction
            recon_err = torch.mean((recon - x) ** 2, dim=1)

            # per-sample KL
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # combined score (negative ELBO)
            score = recon_err + kl
            scores.append(score.cpu().numpy())

    return np.concatenate(scores)


def run_training(subset: str = "FD001"):
    """Main training entry point."""
    cfg = load_config()
    cfg["data"]["dataset"] = subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    data_dir = PROJECT_ROOT / "data" / "raw"
    data = preprocess_subset(
        data_dir,
        subset=subset,
        window_size=cfg["data"]["window_size"],
        stride=cfg["data"]["stride"],
        rul_cap=cfg["data"]["rul_cap"],
    )

    # train all models
    vae_model = train_vae(cfg, data, device)
    ae_model = train_autoencoder(cfg, data, device)
    baselines = train_baselines(cfg, data)

    # save models
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(vae_model.state_dict(), models_dir / f"vae_{subset}.pt")
    torch.save(ae_model.state_dict(), models_dir / f"ae_{subset}.pt")
    logger.info(f"models saved to {models_dir}")

    return {
        "vae": vae_model,
        "autoencoder": ae_model,
        **baselines,
        "data": data,
        "config": cfg,
        "device": device,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    np.random.seed(42)
    torch.manual_seed(42)

    for subset in ["FD001", "FD004"]:
        logger.info(f"\n{'='*60}\nTraining on {subset}\n{'='*60}")
        results = run_training(subset)
