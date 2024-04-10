"""Run full pipeline: train all models, evaluate, generate figures."""

import logging

import numpy as np
import torch

from src.evaluation.anomaly import evaluate_all_models
from src.evaluation.rul import predict_rul_from_latent, predict_rul_from_recon
from src.evaluation.visualization import (
    plot_latent_evolution,
    plot_pr_curves,
    plot_reconstruction_error,
    plot_rul_predictions,
)
from src.models.train import get_reconstruction_scores, run_training

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    all_anomaly = {}
    all_rul = {}

    for subset in ["FD001", "FD004"]:
        logger.info(f"\n{'='*60}\n  {subset}\n{'='*60}")

        results = run_training(subset)
        device = results["device"]
        data = results["data"]

        # anomaly detection
        anomaly_results = evaluate_all_models(results, subset)
        all_anomaly[subset] = anomaly_results

        # PR curves
        plot_pr_curves(anomaly_results, subset)

        # latent space plot: use val engines (run to failure, better trajectories)
        val_data = {
            "test_windows": data["val_windows"],
            "test_rul": data["val_rul"],
            "test_engine_ids": data["val_engine_ids"],
        }
        plot_latent_evolution(results["vae"], val_data, device, subset)

        # recon error vs RUL
        vae_scores = get_reconstruction_scores(
            results["vae"], data["val_windows"], device
        )
        plot_reconstruction_error(vae_scores, data["val_rul"], "VAE", subset)

        # RUL prediction (only for VAE and AE)
        if subset == "FD001":
            vae_rul_recon = predict_rul_from_recon(results["vae"], data, device)
            vae_rul_latent = predict_rul_from_latent(results["vae"], data, device)
            ae_rul = predict_rul_from_recon(results["autoencoder"], data, device)

            all_rul["vae_recon"] = vae_rul_recon
            all_rul["vae_latent"] = vae_rul_latent
            all_rul["ae"] = ae_rul

            plot_rul_predictions(
                vae_rul_latent["ground_truth"],
                vae_rul_latent["predictions"],
                "VAE",
                subset,
            )

    # print summary
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 70)
    for subset, results_list in all_anomaly.items():
        print(f"\n{subset}:")
        for r in results_list:
            print(
                f"  {r['model_name']:20s}  F1={r['best_f1']:.4f}  AP={r['average_precision']:.4f}"
            )

    if all_rul:
        print("\n" + "=" * 70)
        print("RUL PREDICTION (FD001)")
        print("=" * 70)
        for name, r in all_rul.items():
            print(f"  {name:20s}  RMSE={r['rmse']:.2f}  NASA={r['nasa_score']:.0f}")


if __name__ == "__main__":
    main()
