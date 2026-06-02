import os
from pathlib import Path

from . import config


def create_main_folder(username: str) -> dict:
    if not username:
        return {"created": False, "error": "Empty username"}
    path = config.user_dir(username)
    try:
        path.mkdir(parents=True, exist_ok=False)
        return {"created": True, "path": str(path)}
    except FileExistsError:
        return {"created": False, "path": str(path), "error": "exists"}


def create_path_folder(username: str, gesture_number: int, test: str = "") -> dict:
    path = config.gesture_dir(username, gesture_number, test)
    try:
        path.mkdir(parents=True, exist_ok=False)
        return {"created": True, "path": str(path)}
    except FileExistsError:
        return {"created": False, "path": str(path), "error": "exists"}


def list_users() -> list[str]:
    root = config.DATA_ROOT
    if not root.exists():
        return []
    prefix = "G_Alph_"
    return sorted(
        p.name[len(prefix):]
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    )


def list_gestures(username: str) -> list[dict]:
    user_path = config.user_dir(username)
    if not user_path.exists():
        return []
    out = []
    for n, letter in config.GESTURE_MAPPING.items():
        train_path = config.gesture_dir(username, n, "")
        test_path = config.gesture_dir(username, n, "Test")
        trials = 0
        recordings = 0
        if train_path.exists():
            for f in train_path.iterdir():
                if not f.is_file():
                    continue
                if f.name.startswith(f"J{n}.") and f.suffix == ".csv":
                    trials += 1
                elif f.name.startswith(f"{letter}_") and f.suffix == ".csv":
                    # Timestamped raw-backup CSV, written once per "Next gesture" press.
                    recordings += 1
        out.append({
            "n": n,
            "letter": letter,
            "trials": trials,
            "recordings": recordings,
            "has_test": test_path.exists() and (test_path / f"J{n}.1.csv").exists(),
        })
    return out


def gesture_csv_paths(username: str, gesture_number: int, test: str = "") -> list[Path]:
    path = config.gesture_dir(username, gesture_number, test)
    return [path / f"A{i}.csv" for i in range(1, 9)]
