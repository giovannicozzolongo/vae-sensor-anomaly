"""Anomaly detection evaluation."""

import logging

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

from src.models.train import get_reconstruction_scores, run_training

logger = logging.getLogger(__name__)


def label_anomalies(
    rul: np.ndarray, rul_cap: int = 125, degradation_frac: float = 0.3
) -> np.ndarray:
    """Binary labels: 1 = degraded/anomaly, 0 = healthy.

    Window is anomalous if RUL < rul_cap * degradation_frac.
    """
    threshold = rul_cap * degradation_frac
    return (rul < threshold).astype(int)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize anomaly scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn == 0:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def evaluate_anomaly_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Evaluate anomaly scores against binary labels.

    Higher score = more anomalous.
    """
    scores_norm = normalize_scores(scores)

    # best F1 across thresholds
    best_f1 = 0
    best_thresh = 0
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (scores_norm >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores_norm)
    ap = average_precision_score(labels, scores_norm)

    logger.info(
        f"{model_name}: best F1={best_f1:.4f} @ threshold={best_thresh:.2f}, AP={ap:.4f}"
    )

    return {
        "model_name": model_name,
        "best_f1": best_f1,
        "best_threshold": best_thresh,
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "scores": scores_norm,
        "labels": labels,
    }


def evaluate_all_models(results: dict, subset: str = "FD001") -> list[dict]:
    """Run anomaly evaluation on validation engines (full lifecycle → good label balance)."""
    data = results["data"]
    cfg = results["config"]
    device = results["device"]

    # use val_windows  - these engines run to failure so we get both healthy + degraded
    eval_windows = data["val_windows"]
    eval_rul = data["val_rul"]

    labels = label_anomalies(
        eval_rul,
        rul_cap=cfg["data"]["rul_cap"],
        degradation_frac=cfg["evaluation"]["degradation_fraction"],
    )
    logger.info(
        f"anomaly eval: {labels.sum()}/{len(labels)} anomalous ({labels.mean():.2%})"
    )

    all_results = []

    # VAE
    vae_scores = get_reconstruction_scores(results["vae"], eval_windows, device)
    all_results.append(
        evaluate_anomaly_detection(vae_scores, labels, f"VAE ({subset})")
    )

    # AE
    ae_scores = get_reconstruction_scores(results["autoencoder"], eval_windows, device)
    all_results.append(evaluate_anomaly_detection(ae_scores, labels, f"AE ({subset})"))

    # Isolation Forest  - invert scores (lower = more anomalous in sklearn)
    if_scores = -results["isolation_forest"].score_samples(eval_windows)
    all_results.append(evaluate_anomaly_detection(if_scores, labels, f"IF ({subset})"))

    # OC-SVM
    ocsvm_scores = -results["ocsvm"].score_samples(eval_windows)
    all_results.append(
        evaluate_anomaly_detection(ocsvm_scores, labels, f"OC-SVM ({subset})")
    )

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    np.random.seed(42)
    import torch

    torch.manual_seed(42)

    for subset in ["FD001", "FD004"]:
        logger.info(f"\n{'='*60}\nEvaluating {subset}\n{'='*60}")
        train_results = run_training(subset)
        anomaly_results = evaluate_all_models(train_results, subset)

        print(f"\n--- {subset} Anomaly Detection ---")
        for r in anomaly_results:
            print(
                f"  {r['model_name']}: F1={r['best_f1']:.4f}, AP={r['average_precision']:.4f}"
            )
