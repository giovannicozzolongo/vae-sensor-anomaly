"""Visualization: latent space evolution, PR curves, etc."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SensorWindowDataset
from src.models.vae import VAE

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"


def setup_figures_dir():
    FIGURES_DIR.mkdir(exist_ok=True)


def plot_latent_evolution(
    model: VAE,
    data: dict,
    device: torch.device,
    subset: str = "FD001",
    n_engines: int = 5,
    save: bool = True,
):
    """The money plot: latent space trajectories as engines degrade.

    Shows how individual engines move through latent space from healthy to degraded.
    Uses t-SNE or first 2 PCA components if latent_dim > 2.
    """
    setup_figures_dir()
    model.eval()

    windows = data["test_windows"]
    ruls = data["test_rul"]
    eids = data["test_engine_ids"]

    # get latent representations
    ds = SensorWindowDataset(windows, flatten=True)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            z = model.get_latent(x)
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents)

    # PCA to 2D if needed
    if latents.shape[1] > 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        axis_labels = ("PC1", "PC2")
        var_explained = pca.explained_variance_ratio_
        logger.info(
            f"PCA variance explained: {var_explained[0]:.2f}, {var_explained[1]:.2f}"
        )
    else:
        latents_2d = latents
        axis_labels = ("z1", "z2")

    unique_engines = np.unique(eids)
    selected = unique_engines[:n_engines]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # background: all points colored by RUL
    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=ruls,
        cmap="RdYlGn",
        alpha=0.15,
        s=5,
        vmin=0,
        vmax=125,
    )

    # overlay selected engines as trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    for i, eid in enumerate(selected):
        mask = eids == eid
        engine_latents = latents_2d[mask]

        ax.plot(
            engine_latents[:, 0],
            engine_latents[:, 1],
            color=colors[i],
            linewidth=1.5,
            alpha=0.8,
        )
        # mark start and end
        ax.scatter(
            engine_latents[0, 0],
            engine_latents[0, 1],
            color=colors[i],
            marker="o",
            s=60,
            zorder=5,
            edgecolors="black",
        )
        ax.scatter(
            engine_latents[-1, 0],
            engine_latents[-1, 1],
            color=colors[i],
            marker="X",
            s=80,
            zorder=5,
            edgecolors="black",
        )
        ax.annotate(
            f"eng {eid}",
            (engine_latents[0, 0], engine_latents[0, 1]),
            fontsize=7,
            alpha=0.7,
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("RUL (cycles)")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(f"Latent Space Evolution  - {subset}")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"latent_evolution_{subset}.png"
        fig.savefig(path, dpi=100)
        logger.info(f"saved {path}")
    plt.close(fig)


def plot_pr_curves(
    anomaly_results: list[dict],
    subset: str = "FD001",
    save: bool = True,
):
    """Plot precision-recall curves for all models."""
    setup_figures_dir()

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in anomaly_results:
        ax.plot(
            r["recall"],
            r["precision"],
            label=f"{r['model_name']} (AP={r['average_precision']:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves  - {subset}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"pr_curves_{subset}.png"
        fig.savefig(path, dpi=100)
        logger.info(f"saved {path}")
    plt.close(fig)


def plot_reconstruction_error(
    scores: np.ndarray,
    rul: np.ndarray,
    model_name: str = "VAE",
    subset: str = "FD001",
    save: bool = True,
):
    """Scatter plot of reconstruction error vs RUL."""
    setup_figures_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(rul, scores, alpha=0.3, s=3, c=rul, cmap="RdYlGn")
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title(f"{model_name} Reconstruction Error vs RUL  - {subset}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"recon_error_{model_name.lower()}_{subset}.png"
        fig.savefig(path, dpi=100)
        logger.info(f"saved {path}")
    plt.close(fig)


def plot_training_loss(losses: list[float], model_name: str = "VAE", save: bool = True):
    """Simple training loss curve."""
    setup_figures_dir()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} Training Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"training_loss_{model_name.lower()}.png"
        fig.savefig(path, dpi=100)
        logger.info(f"saved {path}")
    plt.close(fig)


def plot_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "VAE",
    subset: str = "FD001",
    save: bool = True,
):
    """Predicted vs actual RUL."""
    setup_figures_dir()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.2, s=5)
    lims = [0, max(y_true.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"RUL Prediction  - {model_name} ({subset})")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"rul_pred_{model_name.lower()}_{subset}.png"
        fig.savefig(path, dpi=100)
        logger.info(f"saved {path}")
    plt.close(fig)
