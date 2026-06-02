"""Two feature sets the rest of the app can train against.

basic  (8-D)   — the legacy E1..E8 energy row read straight from a J-file.
                 One sample per J-file. This is what GloveSense10 trains on.
engineered (48-D)
               — 6 statistics × 8 sensors, computed from each gesture's
                 A1..A8.csv processed-signal files. One sample per gesture
                 recording. Captures the temporal shape the energy summary
                 throws away.

Statistics per sensor channel:
    peak           max(abs(signal))
    std            np.std(signal)
    auc            sum(abs(signal))                        (area under curve)
    slope_max      max(abs(diff(signal)))
    time_to_peak   argmax(abs(signal)) / len(signal)
    fft_dominant   bin index of the max-magnitude FFT bin / len(fft)
                   (skips DC; normalized 0–1)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


N_BASIC = 8
N_ENGINEERED = 48
SENSORS = 8


def _read_signal_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([])
    arr = pd.read_csv(path, header=None).values.ravel()
    return np.array(arr, dtype=float)


def extract_engineered_from_signals(signals) -> np.ndarray:
    """signals: iterable of 8 float arrays (variable lengths allowed). Returns shape (48,)."""
    feats = np.zeros((SENSORS, 6), dtype=float)
    for i, s_raw in enumerate(signals):
        s = np.asarray(s_raw, dtype=float)
        if s.size == 0:
            continue
        absS = np.abs(s)
        feats[i, 0] = float(np.max(absS))
        feats[i, 1] = float(np.std(s))
        feats[i, 2] = float(np.sum(absS))
        diffs = np.diff(s)
        feats[i, 3] = float(np.max(np.abs(diffs))) if diffs.size else 0.0
        feats[i, 4] = float(np.argmax(absS)) / max(1, s.size - 1)
        # FFT dominant bin (skip DC, normalize 0-1)
        if s.size >= 4:
            mags = np.abs(np.fft.rfft(s - np.mean(s)))
            if mags.size > 1:
                dominant_idx = int(np.argmax(mags[1:])) + 1
                feats[i, 5] = dominant_idx / max(1, mags.size - 1)
        else:
            feats[i, 5] = 0.0
    return feats.flatten()


def engineered_for_gesture(gesture_dir: Path) -> np.ndarray | None:
    """Build a single 48-D feature vector for one gesture recording's folder.
    Returns None if the A1..A8 files are missing."""
    signals = []
    for i in range(1, 9):
        s = _read_signal_csv(gesture_dir / f"A{i}.csv")
        if s.size == 0:
            return None
        signals.append(s)
    return extract_engineered_from_signals(signals)


def basic_samples_for_gesture(gesture_dir: Path, gesture_id: int) -> list[np.ndarray]:
    """Return one 8-D feature vector per J-file (variable count: v10→10, v5→5)."""
    out = []
    for fname in sorted(gesture_dir.glob(f"J{gesture_id}.*.csv")):
        df = pd.read_csv(fname)
        if df.shape[1] > 0:
            out.append(df.values.flatten())
    return out


def collect_user(username: str, root: Path | None, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Walk a user's gesture folders under `root` and return (X, y) for one
    feature_set ('basic' or 'engineered'). When root is None, falls back to
    config.DATA_ROOT."""
    base = (root or config.DATA_ROOT) / f"G_Alph_{username}"
    X: list[np.ndarray] = []
    y: list[int] = []
    if not base.exists():
        return np.array([]), np.array([])
    for gid in config.GESTURE_MAPPING:
        gfolder = base / f"Gesture {gid}"
        if not gfolder.exists():
            continue
        if feature_set == "basic":
            for vec in basic_samples_for_gesture(gfolder, gid):
                X.append(vec)
                y.append(gid)
        elif feature_set == "engineered":
            vec = engineered_for_gesture(gfolder)
            if vec is not None:
                X.append(vec)
                y.append(gid)
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
    if not X:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)


def collect_test_sample(username: str, gesture_id: int, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    base = config.gesture_dir(username, gesture_id, "Test")
    if not base.exists():
        return np.array([]), np.array([])
    if feature_set == "basic":
        path = base / f"J{gesture_id}.1.csv"
        if not path.exists():
            return np.array([]), np.array([])
        df = pd.read_csv(path)
        return np.array([df.values.flatten()]), np.array([gesture_id])
    if feature_set == "engineered":
        vec = engineered_for_gesture(base)
        if vec is None:
            return np.array([]), np.array([])
        return np.array([vec]), np.array([gesture_id])
    raise ValueError(f"Unknown feature_set: {feature_set}")
