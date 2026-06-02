"""Helpers used by the collector to write per-trial energy CSVs and the
matplotlib snapshot. Training/inference now lives in training.py."""

import csv

import numpy as np
import pandas as pd

from . import config


def save_energy_csv(username: str, gesture_number: int, test: str, trial: int, energy_row: list[float]) -> str:
    path = config.gesture_dir(username, gesture_number, test) / f"J{gesture_number}.{trial}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"])
        wr.writerow(energy_row)
    return str(path)


def compute_energy_features(processed_signals: np.ndarray, VM: np.ndarray, num_windows: int) -> np.ndarray:
    """Given the 8xN processed-signal matrix and VM index array, return num_windows x 8 energy rows."""
    energies = np.zeros((num_windows, 8))
    for sa in range(num_windows):
        for i in range(8):
            energies[sa, i] = np.sum(processed_signals[i, VM[sa * 5:(sa + 1) * 5]] ** 2)
    return energies


def normalize_energy_rows(energies: np.ndarray) -> np.ndarray:
    """Each column (sample window) normalized by its column max across the 8 sensors."""
    Q = energies.T
    out = np.zeros_like(Q, dtype=float)
    for col in range(Q.shape[1]):
        mx = np.max(Q[:, col])
        out[:, col] = Q[:, col] / mx if mx != 0 else 0.0
    return out


def save_raw_backup(username: str, gesture_number: int, gesture_letter: str, sensors: list[list[int]]) -> str:
    """Save the raw a1..a8 columns as one CSV named like GloveSense10 does."""
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = config.gesture_dir(username, gesture_number, "")
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{gesture_letter}_{ts}.csv"
    df = pd.DataFrame({f"a{i + 1}": sensors[i] for i in range(8)})
    df.to_csv(file_path, index=False)
    return str(file_path)
