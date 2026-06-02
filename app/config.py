from pathlib import Path

SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600
SERIAL_TIMEOUT = 1

# Matches GloveSense10.py's TRAIN_ROOT = "../GloveSenseDash/"
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "GloveSenseDash"
GESTURES_IMG_DIR = Path(__file__).resolve().parent.parent / "gestures"

# Extra read-only directories scanned by evaluation routines (LOOU, leaderboard).
# Use it to pull in the legacy reference dataset that ships with the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_DIRS = [DATA_ROOT, _REPO_ROOT / "Data and scripts"]

# Test collection is identical across variants
TEST_SAMPLES_PER_SENSOR = 15
TEST_WINDOWS = 1
TEST_DATA_STREAM = 10

SENSOR_COUNT = 8
WINDOW_SIZE = 5

# Per-variant training collection. Each variant maps to a target sample count,
# a data-stream window used by signal processing, and the number of J-files
# produced. Matches GloveSense10.py (v10) and GloveSense5.py (v5).
VARIANTS = {
    "v10": {"label": "10 reps", "train_samples": 105, "data_stream": 100, "train_windows": 10},
    "v5":  {"label": "5 reps",  "train_samples":  55, "data_stream":  50, "train_windows":  5},
}
DEFAULT_VARIANT = "v10"

# Back-compat shims (used by older code paths during the transition)
TRAIN_SAMPLES_PER_SENSOR = VARIANTS[DEFAULT_VARIANT]["train_samples"]
TRAIN_WINDOWS = VARIANTS[DEFAULT_VARIANT]["train_windows"]

GESTURE_MAPPING = {
    1: "A", 2: "B", 3: "C", 4: "D",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "L",
    10: "N", 11: "W", 12: "Y", 13: "TR", 14: "TIM", 15: "TMR",
}

REST_IMAGE_ID = 0
SMILE_IMAGE = "smile.jpg"
SAD_IMAGE = "sad.jpg"


def user_dir(username: str) -> Path:
    return DATA_ROOT / f"G_Alph_{username}"


def gesture_dir(username: str, gesture_number: int, test: str = "") -> Path:
    return user_dir(username) / f"Gesture {gesture_number}{test}"


def model_dir(username: str) -> Path:
    return user_dir(username) / "model"


def rf_model_path(username: str) -> Path:
    return model_dir(username) / "rf_model.pkl"


def label_encoder_path(username: str) -> Path:
    return model_dir(username) / "label_encoder.pkl"
