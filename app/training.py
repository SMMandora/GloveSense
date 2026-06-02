"""Per-user training and prediction with the multi-model zoo.

train_full(username, feature_set) — fits every available model on the user's
data, saves each under model/<model_name>__<feature_set>.pkl plus a shared
label_encoder__<feature_set>.pkl. Returns each model's training accuracy.

predict_user(username, gesture_number, model_name, feature_set) — loads the
matching saved estimator + encoder and runs a single-sample prediction
against the gesture's Test/J{n}.1.csv (or engineered A1..A8 stack).
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from . import config
from .feature_extraction import collect_test_sample, collect_user
from .models import (
    DEFAULT_MODEL,
    available_models,
    build,
    display_name,
)


def _model_path(username: str, model_name: str, feature_set: str) -> Path:
    return config.model_dir(username) / f"{model_name}__{feature_set}.pkl"


def _label_encoder_path(username: str, feature_set: str) -> Path:
    return config.model_dir(username) / f"label_encoder__{feature_set}.pkl"


def train_full(username: str, feature_set: str = "basic", models: list[str] | None = None) -> dict:
    if not username:
        raise ValueError("username required")
    X, y = collect_user(username, config.DATA_ROOT, feature_set)
    if X.size == 0 or y.size == 0:
        raise FileNotFoundError("No training data found. Collect gestures first.")

    config.model_dir(username).mkdir(parents=True, exist_ok=True)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, _label_encoder_path(username, feature_set))

    selected = models or available_models()
    results: list[dict] = []
    for name in selected:
        try:
            est = build(name)
            est.fit(X, y_encoded)
            path = _model_path(username, name, feature_set)
            joblib.dump(est, path)
            acc = float(accuracy_score(y_encoded, est.predict(X)))
            results.append({
                "model": name,
                "display": display_name(name),
                "train_accuracy": acc,
                "path": str(path),
            })
        except Exception as e:  # one bad model shouldn't kill the others
            results.append({
                "model": name,
                "display": display_name(name),
                "error": str(e),
            })

    # Keep the legacy rf_model.pkl path populated so GloveSense10-style consumers still work.
    if feature_set == "basic" and "rf" in selected:
        try:
            joblib.dump(joblib.load(_model_path(username, "rf", "basic")), config.rf_model_path(username))
            joblib.dump(le, config.label_encoder_path(username))
        except Exception:
            pass

    return {
        "feature_set": feature_set,
        "samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "classes_trained": [config.GESTURE_MAPPING.get(int(c), str(c)) for c in le.classes_],
        "results": results,
        "label_encoder_path": str(_label_encoder_path(username, feature_set)),
    }


def predict_user(
    username: str,
    gesture_number: int,
    model_name: str = DEFAULT_MODEL,
    feature_set: str = "basic",
) -> dict:
    if not username:
        raise ValueError("username required")
    model_path = _model_path(username, model_name, feature_set)
    le_path = _label_encoder_path(username, feature_set)
    if not model_path.exists() or not le_path.exists():
        raise FileNotFoundError(
            f"Trained {display_name(model_name)} ({feature_set}) not found. Train first."
        )

    X_test, y_test = collect_test_sample(username, gesture_number, feature_set)
    if X_test.size == 0:
        raise FileNotFoundError(f"No test data found for gesture {gesture_number}.")

    le = joblib.load(le_path)
    est = joblib.load(model_path)

    y_pred_encoded = est.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)
    expected = config.GESTURE_MAPPING.get(int(y_test[0]), "Unknown")
    predicted = config.GESTURE_MAPPING.get(int(y_pred[0]), "Unknown")
    correct = bool(int(y_pred[0]) == int(y_test[0]))

    probs = None
    try:
        if hasattr(est, "predict_proba"):
            raw = est.predict_proba(X_test)[0]
            probs = sorted(
                (
                    {
                        "letter": config.GESTURE_MAPPING.get(int(le.inverse_transform([i])[0]), str(i)),
                        "p": float(p),
                    }
                    for i, p in enumerate(raw)
                ),
                key=lambda d: d["p"],
                reverse=True,
            )[:5]
    except Exception:
        probs = None

    return {
        "expected": expected,
        "expected_id": int(y_test[0]),
        "predicted": predicted,
        "predicted_id": int(y_pred[0]),
        "correct": correct,
        "model": model_name,
        "model_display": display_name(model_name),
        "feature_set": feature_set,
        "feature_vector": X_test[0].tolist(),
        "top_probabilities": probs,
    }
