"""Preprocessing for C-MAPSS turbofan data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# constant in FD001/FD003 (single operating condition)
ZERO_VAR_SENSORS_FD001 = [1, 5, 6, 10, 16, 18, 19]


def load_raw(
    data_dir: str | Path, subset: str = "FD001"
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load raw train, test, and RUL files for a given subset."""
    data_dir = Path(data_dir)

    train = pd.read_csv(
        data_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test = pd.read_csv(
        data_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul = pd.read_csv(
        data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul"]
    )["rul"].values

    logger.info(
        f"loaded {subset}: train={len(train)}, test={len(test)}, engines={train.engine_id.nunique()}"
    )
    return train, test, rul


def add_rul_labels(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """Add piece-wise linear RUL labels capped at rul_cap."""
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max()

    rul_vals = []
    for _, row in df.iterrows():
        max_c = max_cycles[row["engine_id"]]
        rul = max_c - row["cycle"]
        rul_vals.append(min(rul, rul_cap))

    df["rul"] = rul_vals
    return df


def drop_zero_variance(df: pd.DataFrame, subset: str = "FD001") -> pd.DataFrame:
    """Drop sensors with zero variance for single-condition subsets."""
    if subset in ("FD001", "FD003"):
        cols_to_drop = [f"sensor_{i}" for i in ZERO_VAR_SENSORS_FD001]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"dropped {len(cols_to_drop)} zero-variance sensors for {subset}")
    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def normalize_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize sensor readings per engine."""
    df = df.copy()
    sensor_cols = get_sensor_columns(df)
    df[sensor_cols] = df[sensor_cols].astype(np.float64)

    for eid in df.engine_id.unique():
        mask = df.engine_id == eid
        engine_data = df.loc[mask, sensor_cols]
        mins = engine_data.min()
        maxs = engine_data.max()
        rng = maxs - mins
        rng[rng == 0] = 1.0
        df.loc[mask, sensor_cols] = (engine_data - mins) / rng

    return df


def create_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
    sensor_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows from the dataframe.

    Returns:
        windows: (n_windows, window_size, n_features)
        rul_labels: RUL at the end of each window
        engine_ids: engine ID for each window
    """
    if sensor_cols is None:
        sensor_cols = get_sensor_columns(df)

    windows, ruls, eids = [], [], []

    for eid in df.engine_id.unique():
        engine = df[df.engine_id == eid].sort_values("cycle")
        values = engine[sensor_cols].values
        rul_values = engine["rul"].values if "rul" in engine.columns else None

        n_steps = len(values)
        if n_steps < window_size:
            continue

        for start in range(0, n_steps - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end])
            eids.append(eid)
            if rul_values is not None:
                ruls.append(rul_values[end - 1])

    windows = np.array(windows, dtype=np.float32)
    engine_ids = np.array(eids)
    rul_labels = np.array(ruls, dtype=np.float32) if ruls else np.array([])

    logger.info(f"created {len(windows)} windows of shape {windows.shape[1:]}")
    return windows, rul_labels, engine_ids


def preprocess_subset(
    data_dir: str | Path,
    subset: str = "FD001",
    window_size: int = 30,
    stride: int = 1,
    rul_cap: int = 125,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Full preprocessing pipeline for one subset.

    Splits training engines into train/val. Val engines run to failure,
    so they have both healthy and degraded windows  - good for anomaly eval.
    Test set is used for RUL prediction.
    """
    train_df, test_df, test_rul = load_raw(data_dir, subset)

    # add RUL to train (run to failure)
    train_df = add_rul_labels(train_df, rul_cap=rul_cap)

    # add RUL to test using ground truth file
    test_df = _add_test_rul(test_df, test_rul, rul_cap=rul_cap)

    # drop zero-variance sensors
    train_df = drop_zero_variance(train_df, subset)
    test_df = drop_zero_variance(test_df, subset)

    sensor_cols = get_sensor_columns(train_df)

    # split training engines into train/val
    all_engines = np.sort(train_df.engine_id.unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_engines)
    n_val = max(1, int(len(all_engines) * val_fraction))
    val_engines = set(all_engines[:n_val])
    train_engines = set(all_engines[n_val:])

    train_split = train_df[train_df.engine_id.isin(train_engines)]
    val_split = train_df[train_df.engine_id.isin(val_engines)]

    logger.info(
        f"split: {len(train_engines)} train engines, {len(val_engines)} val engines"
    )

    # normalize per engine
    train_split = normalize_per_engine(train_split)
    val_split = normalize_per_engine(val_split)
    test_df = normalize_per_engine(test_df)

    # windows
    train_windows, train_ruls, train_eids = create_windows(
        train_split, window_size, stride, sensor_cols
    )
    val_windows, val_ruls, val_eids = create_windows(
        val_split, window_size, stride, sensor_cols
    )
    test_windows, test_ruls, test_eids = create_windows(
        test_df, window_size, stride, sensor_cols
    )

    return {
        "train_windows": train_windows,
        "train_rul": train_ruls,
        "train_engine_ids": train_eids,
        "val_windows": val_windows,
        "val_rul": val_ruls,
        "val_engine_ids": val_eids,
        "test_windows": test_windows,
        "test_rul": test_ruls,
        "test_engine_ids": test_eids,
        "sensor_cols": sensor_cols,
        "subset": subset,
    }


def _add_test_rul(
    test_df: pd.DataFrame, rul_values: np.ndarray, rul_cap: int = 125
) -> pd.DataFrame:
    """Add RUL column to test data using ground truth."""
    test_df = test_df.copy()
    rul_list = []
    for eid in test_df.engine_id.unique():
        engine = test_df[test_df.engine_id == eid].sort_values("cycle")
        max_cycle = engine.cycle.max()
        remaining = rul_values[eid - 1]  # 0-indexed
        for _, row in engine.iterrows():
            r = remaining + (max_cycle - row["cycle"])
            rul_list.append(min(r, rul_cap))
    test_df["rul"] = rul_list
    return test_df
