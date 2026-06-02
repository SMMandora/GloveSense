"""Single source of truth for the model zoo. Each model returns a sklearn-compatible
estimator (or Pipeline) ready to .fit / .predict on (X, y_encoded) — with scaling
folded in where it matters.

Model keys:
    rf       Random Forest          — robust baseline, no scaling
    lgbm     LightGBM               — typically best on low-dim tabular at this size
    svm      SVC (RBF)              — strong fit for 8-D / 48-D feature spaces
    knn      K-Nearest Neighbors    — sanity baseline; scales well at low-dim
    ensemble Soft-voting ensemble   — RF + LightGBM + SVM (uncorrelated errors)
"""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False


MODEL_KEYS = ["rf", "lgbm", "svm", "knn", "ensemble"]
DEFAULT_MODEL = "rf"


def _rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )


def _lgbm():
    if not HAS_LGBM:
        raise RuntimeError("LightGBM not installed.")
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )


def _svm():
    # SVC needs scaled features; bake the scaler into the model.
    return Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(
            kernel="rbf", C=4.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=42,
        )),
    ])


def _knn():
    return Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
    ])


def _ensemble():
    estimators = [("rf", _rf()), ("svm", _svm())]
    if HAS_LGBM:
        estimators.insert(1, ("lgbm", _lgbm()))
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)


_FACTORIES = {
    "rf": _rf,
    "lgbm": _lgbm,
    "svm": _svm,
    "knn": _knn,
    "ensemble": _ensemble,
}


def available_models() -> list[str]:
    """Return the keys that can actually be built in this environment."""
    out = []
    for k in MODEL_KEYS:
        if k == "lgbm" and not HAS_LGBM:
            continue
        out.append(k)
    return out


def build(name: str):
    if name not in _FACTORIES:
        raise ValueError(f"Unknown model: {name}")
    return _FACTORIES[name]()


def display_name(name: str) -> str:
    return {
        "rf": "Random Forest",
        "lgbm": "LightGBM",
        "svm": "SVM (RBF)",
        "knn": "k-NN (k=5)",
        "ensemble": "Voting Ensemble",
    }.get(name, name)
