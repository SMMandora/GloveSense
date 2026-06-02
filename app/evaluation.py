"""Cross-validation leaderboard + leave-one-user-out evaluation.

leaderboard(feature_set) — pool every user across config.EVAL_DATA_DIRS,
                            run 5-fold stratified CV across every model.

loou(feature_set)        — for each user, train on the other users, test on
                            the held-out user. Reports per-user accuracy +
                            an aggregate.

confusion(feature_set, model_name) — fit the requested model on the pooled
                                      data and return a confusion matrix.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from . import config
from .feature_extraction import collect_user
from .models import available_models, build, display_name


def _has_gesture_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.is_dir() and child.name.startswith("Gesture ") for child in path.iterdir())


def _resolve_user_root(outer: Path) -> Path | None:
    """Some legacy trials were re-zipped doubly-nested as G_Alph_X/G_Alph_X/Gesture *.
    Walk up to two levels deep looking for the level that actually holds Gesture * subdirs."""
    if _has_gesture_subdirs(outer):
        return outer
    for child in outer.iterdir():
        if child.is_dir() and (child.name == outer.name or child.name.startswith("G_Alph_")):
            if _has_gesture_subdirs(child):
                return child
    return None


def _discover_users() -> list[tuple[str, Path]]:
    """Return (username, root_dir) for every G_Alph_* across configured dirs,
    accounting for the doubly-nested legacy layout."""
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for src in config.EVAL_DATA_DIRS:
        if not src.exists():
            continue
        for p in sorted(src.iterdir()):
            if not (p.is_dir() and p.name.startswith("G_Alph_")):
                continue
            resolved = _resolve_user_root(p)
            if resolved is None:
                continue
            name = p.name[len("G_Alph_"):]
            if name in seen:
                continue
            seen.add(name)
            # parent_of_resolved is what collect_user joins back with "G_Alph_{name}".
            # When doubly-nested, resolved is .../G_Alph_X/G_Alph_X — its parent is the
            # outer G_Alph_X folder, so collect_user(name, parent_of_resolved.parent)
            # would re-look up G_Alph_X under the OUTER outer. So we pass resolved.parent
            # directly so that resolved.parent / f"G_Alph_{name}" == resolved.
            out.append((name, resolved.parent))
    return out


def _pool(feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every user's (X, y) plus a parallel `groups` array tagging
    each sample with its user index (for LOOU)."""
    Xs, ys, gs = [], [], []
    for idx, (user, src) in enumerate(_discover_users()):
        X, y = collect_user(user, src, feature_set)
        if X.size == 0:
            continue
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(y), idx, dtype=int))
    if not Xs:
        return np.array([]), np.array([]), np.array([])
    return np.vstack(Xs), np.hstack(ys), np.hstack(gs)


def leaderboard(feature_set: str, n_splits: int = 5, models: list[str] | None = None) -> dict:
    X, y, _ = _pool(feature_set)
    if X.size == 0:
        return {"feature_set": feature_set, "users": 0, "samples": 0, "results": []}

    # Smallest per-class count (ignore unused class ids like 0 if labels are 1..15)
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.min()))
    if n_splits < 2:
        return {
            "feature_set": feature_set,
            "users": len(_discover_users()),
            "samples": int(X.shape[0]),
            "error": "not enough samples per class for k-fold CV",
            "results": [],
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    selected = models or available_models()
    results: list[dict] = []
    for name in selected:
        fold_accs: list[float] = []
        err: str | None = None
        for train_idx, test_idx in cv.split(X, y):
            try:
                est = build(name)
                est.fit(X[train_idx], y[train_idx])
                fold_accs.append(float(accuracy_score(y[test_idx], est.predict(X[test_idx]))))
            except Exception as e:
                err = str(e)
                break
        if err is not None:
            results.append({"model": name, "display": display_name(name), "error": err})
        else:
            results.append({
                "model": name,
                "display": display_name(name),
                "cv_accuracy_mean": float(np.mean(fold_accs)),
                "cv_accuracy_std": float(np.std(fold_accs)),
                "folds": fold_accs,
            })
    results.sort(key=lambda d: d.get("cv_accuracy_mean", -1), reverse=True)
    return {
        "feature_set": feature_set,
        "users": len(_discover_users()),
        "samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "n_splits": n_splits,
        "results": results,
    }


def loou(feature_set: str, models: list[str] | None = None) -> dict:
    X, y, groups = _pool(feature_set)
    if X.size == 0:
        return {"feature_set": feature_set, "users": 0, "samples": 0, "results": []}

    user_names = [name for (name, _src) in _discover_users()]
    selected = models or available_models()
    per_model: dict[str, dict] = {}

    for name in selected:
        per_user: list[dict] = []
        for gid in range(len(user_names)):
            mask_train = groups != gid
            mask_test = groups == gid
            if not mask_test.any() or not mask_train.any():
                continue
            try:
                est = build(name)
                est.fit(X[mask_train], y[mask_train])
                acc = float(accuracy_score(y[mask_test], est.predict(X[mask_test])))
                per_user.append({"user": user_names[gid], "accuracy": acc, "samples": int(mask_test.sum())})
            except Exception as e:
                per_user.append({"user": user_names[gid], "error": str(e), "samples": int(mask_test.sum())})
        accs = [u["accuracy"] for u in per_user if "accuracy" in u]
        per_model[name] = {
            "display": display_name(name),
            "mean": float(np.mean(accs)) if accs else None,
            "std": float(np.std(accs)) if accs else None,
            "per_user": per_user,
        }

    ranked = sorted(
        [{"model": k, **v} for k, v in per_model.items()],
        key=lambda d: d.get("mean") or -1,
        reverse=True,
    )
    return {
        "feature_set": feature_set,
        "users": len(user_names),
        "user_names": user_names,
        "samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "results": ranked,
    }


def confusion(feature_set: str, model_name: str) -> dict:
    X, y, _ = _pool(feature_set)
    if X.size == 0:
        return {"matrix": [], "labels": []}
    est = build(model_name)
    est.fit(X, y)
    y_pred = est.predict(X)
    labels = sorted(set(int(v) for v in y))
    mat = confusion_matrix(y, y_pred, labels=labels)
    return {
        "matrix": mat.tolist(),
        "labels": [config.GESTURE_MAPPING.get(l, str(l)) for l in labels],
        "label_ids": labels,
        "samples": int(X.shape[0]),
        "model": model_name,
        "model_display": display_name(model_name),
        "feature_set": feature_set,
    }
