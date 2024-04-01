"""Baseline anomaly detection: Isolation Forest and One-Class SVM."""

import logging

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Wraps sklearn IsolationForest for our pipeline."""

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray):
        """Fit on healthy/normal data (flattened windows)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X)
        logger.info(f"IF fitted on {len(X)} samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for normal, -1 for anomaly."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores  - lower = more anomalous."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.score_samples(X)


class OCSVMDetector:
    """One-class SVM wrapper."""

    def __init__(self, kernel: str = "rbf", nu: float = 0.1, gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X: np.ndarray):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        # subsample if too large, OC-SVM doesn't scale well
        if len(X) > 5000:
            idx = np.random.RandomState(42).choice(len(X), 5000, replace=False)
            X = X[idx]
            logger.info("subsampled to 5000 for OC-SVM fitting")
        self.model.fit(X)
        logger.info(f"OC-SVM fitted on {len(X)} samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.decision_function(X)
