"""RUL prediction and evaluation using reconstruction error as a proxy."""

import logging

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

from src.data.dataset import SensorWindowDataset
from src.models.train import get_reconstruction_scores, run_training
from src.models.vae import VAE

logger = logging.getLogger(__name__)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA asymmetric scoring function.

    Penalizes late predictions (under-estimation) more than early ones.
    d = predicted - actual:
        d < 0 (early): exp(-d/13) - 1
        d >= 0 (late): exp(d/10) - 1
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _get_last_window_per_engine(
    windows: np.ndarray,
    rul: np.ndarray,
    engine_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the last window of each engine for RUL evaluation."""
    unique_engines = np.unique(engine_ids)
    last_idx = []
    for eid in unique_engines:
        mask = engine_ids == eid
        indices = np.where(mask)[0]
        last_idx.append(indices[-1])
    last_idx = np.array(last_idx)
    return windows[last_idx], rul[last_idx], unique_engines


def _get_latents(model: VAE, windows: np.ndarray, device: torch.device) -> np.ndarray:
    from torch.utils.data import DataLoader

    model.eval()
    ds = SensorWindowDataset(windows, flatten=True)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            z = model.get_latent(x)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents)


def predict_rul_from_latent(
    model: VAE,
    data: dict,
    device: torch.device,
) -> dict:
    """Predict RUL from latent features + recon error using gradient boosting.

    Fits on training+val data, evaluates per-engine on test.
    """
    # combine train + val for fitting (both run to failure)
    all_windows = np.concatenate([data["train_windows"], data["val_windows"]])
    all_rul = np.concatenate([data["train_rul"], data["val_rul"]])

    all_z = _get_latents(model, all_windows, device)
    all_scores = get_reconstruction_scores(model, all_windows, device)
    all_features = np.column_stack([all_z, all_scores])

    # subsample for speed
    rng = np.random.RandomState(42)
    n = len(all_features)
    if n > 5000:
        idx = rng.choice(n, 5000, replace=False)
        fit_features = all_features[idx]
        fit_rul = all_rul[idx]
    else:
        fit_features = all_features
        fit_rul = all_rul

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    reg.fit(fit_features, fit_rul)

    # test: get last window per engine
    test_w, test_rul_last, test_eids = _get_last_window_per_engine(
        data["test_windows"], data["test_rul"], data["test_engine_ids"]
    )
    test_z = _get_latents(model, test_w, device)
    test_scores = get_reconstruction_scores(model, test_w, device)
    test_features = np.column_stack([test_z, test_scores])

    pred = np.clip(reg.predict(test_features), 0, 125)

    rul_rmse = rmse(test_rul_last, pred)
    rul_nasa = nasa_score(test_rul_last, pred)
    logger.info(
        f"VAE (latent+recon) per-engine: RMSE={rul_rmse:.2f}, NASA={rul_nasa:.0f}"
    )

    return {
        "rmse": rul_rmse,
        "nasa_score": rul_nasa,
        "predictions": pred,
        "ground_truth": test_rul_last,
    }


def predict_rul_direct(data: dict) -> dict:
    """Predict RUL directly from windowed sensor features using gradient boosting.

    Uses last window statistics as features.
    """
    # flatten windows for regression
    all_windows = np.concatenate([data["train_windows"], data["val_windows"]])
    all_rul = np.concatenate([data["train_rul"], data["val_rul"]])

    n, ws, nf = all_windows.shape
    # use last-timestep values + mean + std as features
    all_feat = np.column_stack(
        [
            all_windows[:, -1, :],  # last timestep
            all_windows.mean(axis=1),  # mean over window
            all_windows.std(axis=1),  # std over window
            all_windows[:, -1, :] - all_windows[:, 0, :],  # delta first-to-last
        ]
    )

    rng = np.random.RandomState(42)
    if len(all_feat) > 5000:
        idx = rng.choice(len(all_feat), 5000, replace=False)
        fit_feat = all_feat[idx]
        fit_rul = all_rul[idx]
    else:
        fit_feat = all_feat
        fit_rul = all_rul

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    reg.fit(fit_feat, fit_rul)

    # test: last window per engine
    test_w, test_rul_last, _ = _get_last_window_per_engine(
        data["test_windows"], data["test_rul"], data["test_engine_ids"]
    )
    test_feat = np.column_stack(
        [
            test_w[:, -1, :],
            test_w.mean(axis=1),
            test_w.std(axis=1),
            test_w[:, -1, :] - test_w[:, 0, :],
        ]
    )

    pred = np.clip(reg.predict(test_feat), 0, 125)
    rul_rmse = rmse(test_rul_last, pred)
    rul_nasa = nasa_score(test_rul_last, pred)
    logger.info(
        f"direct (sensor features) per-engine: RMSE={rul_rmse:.2f}, NASA={rul_nasa:.0f}"
    )

    return {
        "rmse": rul_rmse,
        "nasa_score": rul_nasa,
        "predictions": pred,
        "ground_truth": test_rul_last,
    }


def predict_rul_from_recon(
    model,
    data: dict,
    device: torch.device,
    model_name: str = "model",
) -> dict:
    """Predict RUL from reconstruction error.

    Fits on val data (full lifecycle), predicts on last test window.
    """
    all_windows = np.concatenate([data["train_windows"], data["val_windows"]])
    all_rul = np.concatenate([data["train_rul"], data["val_rul"]])
    all_scores = get_reconstruction_scores(model, all_windows, device)

    reg = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)
    reg.fit(all_scores.reshape(-1, 1), all_rul)

    # last window per test engine
    test_w, test_rul_last, test_eids = _get_last_window_per_engine(
        data["test_windows"], data["test_rul"], data["test_engine_ids"]
    )
    test_scores = get_reconstruction_scores(model, test_w, device)
    pred = np.clip(reg.predict(test_scores.reshape(-1, 1)), 0, 125)

    rul_rmse = rmse(test_rul_last, pred)
    rul_nasa = nasa_score(test_rul_last, pred)
    logger.info(
        f"{model_name} (recon) per-engine: RMSE={rul_rmse:.2f}, NASA={rul_nasa:.0f}"
    )

    return {
        "rmse": rul_rmse,
        "nasa_score": rul_nasa,
        "predictions": pred,
        "ground_truth": test_rul_last,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    np.random.seed(42)
    torch.manual_seed(42)

    for subset in ["FD001"]:
        logger.info(f"\n{'='*60}\nRUL evaluation on {subset}\n{'='*60}")
        results = run_training(subset)

        vae_latent = predict_rul_from_latent(
            results["vae"], results["data"], results["device"]
        )
        vae_recon = predict_rul_from_recon(
            results["vae"], results["data"], results["device"], "VAE"
        )
        ae_recon = predict_rul_from_recon(
            results["autoencoder"], results["data"], results["device"], "AE"
        )
