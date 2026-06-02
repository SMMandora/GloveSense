# Glove Sense — Sign Language Studio

A FastAPI backend + vanilla HTML/CSS/JS frontend that mirrors the behavior of the original `GloveSense10.py` Dash app, with a redesigned UI. Captures sensor data from an 8-channel ESP8266 glove ([PCB-GLOVE/PCB-GLOVE.ino](PCB-GLOVE/PCB-GLOVE.ino)), processes the signal, trains a Random Forest classifier per user, and predicts gestures with a friendly smile/sad reaction.

## What's in the box

- **Polished single-page UI**: bento layout, animated phase ring (REST / HOLD), live progress bar, gesture-progress grid, smile/sad reaction on prediction.
- **WebSocket streaming**: 8 live Chart.js plots updating in real time as samples arrive.
- **Pause / resume / abort** controls during collection.
- **Per-user model** stored at `GloveSenseDash/G_Alph_<user>/model/rf_model.pkl` (+ `label_encoder.pkl`) — same layout as `GloveSense10.py`.

## Project layout

```
app/                    FastAPI backend
  main.py               routes + WebSocket + static mounts
  config.py             COM port, baud, gesture mapping, paths
  serial_io.py          SerialClient singleton (pyserial)
  collector.py          async sample-collection generator
  signal_processing.py  process_signal, fill_outliers
  features.py           energy CSV writers, raw backup, normalization
  training.py           Random Forest train + predict
  storage.py            user/gesture folder management
web/                    Frontend (no build step)
  index.html
  style.css
  app.js
  vendor/chart.umd.min.js
gestures/               Reference images (0.jpg–16.jpg, smile.jpg, sad.jpg)
PCB-GLOVE/              Arduino firmware (unchanged)
Eight_sensors.py        Legacy Streamlit app (reference; delete after on-glove verification)
GloveSense10.py         Legacy Dash app (reference)
```

Data lives in `H:\GloveSenseDash\G_Alph_<user>\` to match `GloveSense10.py`'s `TRAIN_ROOT`.

## Setup

```
pip install -r requirements.txt
```

The Arduino sketch runs at 9600 baud on `COM3`; both are set in [app/config.py](app/config.py).

## Run

```
python run.py
```

Open <http://127.0.0.1:8000/>. Auto-generated API docs at <http://127.0.0.1:8000/docs>.

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/api/health` | Liveness + active config |
| GET    | `/api/gestures` | Index→letter mapping |
| GET    | `/api/users` | List profiles in `GloveSenseDash/` |
| POST   | `/api/users` | Create `G_Alph_<u>/` |
| GET    | `/api/users/{u}/gestures` | Per-gesture trial counts |
| **WS** | `/ws/collect` | Stream live collection. Send `{username, gesture_number, mode}`; receive `started`/`phase`/`sample`/`paused`/`resumed`/`done`/`aborted`/`error`. Send `{action: "pause"\|"resume"\|"abort"}` to control. |
| POST   | `/api/users/{u}/train` | Train RF, save `model/rf_model.pkl` + `model/label_encoder.pkl` |
| POST   | `/api/users/{u}/predict` | Predict the selected gesture from its `Test/J{n}.1.csv`; returns `{expected, predicted, correct}` |
| GET    | `/api/users/{u}/gestures/{n}/data` | Read saved processed signals |

## UI flow

1. **Pick or create a profile** in the left sidebar — folder is created under `GloveSenseDash/`.
2. **Train tab → Next gesture**: cycles through the 15 letters (A, B, C, D, F, G, H, I, L, N, W, Y, TR, TIM, TMR). During collection:
   - The phase ring around the gesture image pulses blue during REST and rose during HOLD.
   - A live progress bar fills as samples come in (target 105 per sensor for training, 15 for testing).
   - Eight Chart.js panels animate sample-by-sample.
   - Pause / Abort always available.
3. **Train Random Forest** trains on every J-file under the user's gesture folders, reports accuracy + sample count.
4. **Test tab**: pick a gesture, **Capture test sample**, then **Run prediction**. Result shows a smile.jpg or sad.jpg card with predicted vs expected letters.

## Hardware contract (matches GloveSense10.py)

- 8 sensors, indexed 1–8 (thumb → ulnar).
- Arduino at 9600 baud on `COM3`.
- Python triggers the next sample by writing `b"g"` to serial; the firmware sends `<sensor_num>\n<value>\n`.
- After 105 samples per channel for training (15 for test), the collector:
  - subtracts the first-5-sample baseline,
  - runs `process_signal` (spike removal, window averaging, VR/VM gating),
  - normalizes per-window across sensors,
  - writes `A1.csv–A8.csv` + 10 `J{n}.{trial}.csv` files (1 J-file for test),
  - and saves a `{letter}_{timestamp}.csv` raw backup.

## Known limitations

- Single serial port → one concurrent collection. A second `/ws/collect` connection receives `{"event": "error", "detail": "serial port busy"}`.
- Server binds to `127.0.0.1` only; no auth.
- Live serial paths require the actual glove on `COM3`. The smoke tests in this repo cover everything else.
