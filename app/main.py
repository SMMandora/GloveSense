import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .collector import collect_gesture
from .evaluation import confusion as eval_confusion
from .evaluation import leaderboard as eval_leaderboard
from .evaluation import loou as eval_loou
from .models import DEFAULT_MODEL, available_models, display_name
from .serial_io import glove_status
from .storage import create_main_folder, list_gestures, list_users
from .training import predict_user, train_full


WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="GloveSense", version="0.2.0")


class CreateUser(BaseModel):
    username: str


class PredictPayload(BaseModel):
    gesture_number: int
    model: str | None = None
    feature_set: str | None = "basic"


class TrainPayload(BaseModel):
    feature_set: str | None = "basic"
    models: list[str] | None = None


class EvalPayload(BaseModel):
    feature_set: str = "basic"
    models: list[str] | None = None


class ConfusionPayload(BaseModel):
    feature_set: str = "basic"
    model: str = DEFAULT_MODEL


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "data_root": str(config.DATA_ROOT),
        "gestures_dir": str(config.GESTURES_IMG_DIR),
        "serial_port": config.SERIAL_PORT,
        "serial_baud": config.SERIAL_BAUD,
    }


@app.get("/api/glove/status")
def get_glove_status() -> dict:
    return glove_status()


@app.get("/api/gestures")
def gestures_mapping() -> dict:
    return {
        "mapping": {str(k): v for k, v in config.GESTURE_MAPPING.items()},
        "variants": config.VARIANTS,
        "default_variant": config.DEFAULT_VARIANT,
    }


@app.get("/api/models")
def get_models() -> dict:
    return {
        "models": [
            {"key": k, "display": display_name(k)} for k in available_models()
        ],
        "feature_sets": [
            {"key": "basic", "display": "Energy (8-D)", "dim": 8},
            {"key": "engineered", "display": "Engineered (48-D)", "dim": 48},
        ],
        "default_model": DEFAULT_MODEL,
        "default_feature_set": "basic",
    }


@app.get("/api/users")
def get_users() -> dict:
    return {"users": list_users()}


@app.post("/api/users")
def post_users(payload: CreateUser) -> dict:
    if not payload.username.strip():
        raise HTTPException(status_code=400, detail="username required")
    result = create_main_folder(payload.username.strip())
    return {"username": payload.username.strip(), **result}


@app.get("/api/users/{username}/gestures")
def get_user_gestures(username: str) -> dict:
    return {"gestures": list_gestures(username)}


@app.post("/api/users/{username}/train")
async def post_train(username: str, payload: TrainPayload | None = None) -> dict:
    feature_set = (payload.feature_set if payload else None) or "basic"
    models = payload.models if payload else None
    try:
        return await asyncio.to_thread(train_full, username, feature_set, models)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/users/{username}/predict")
async def post_predict(username: str, payload: PredictPayload) -> dict:
    model = payload.model or DEFAULT_MODEL
    feature_set = payload.feature_set or "basic"
    try:
        return await asyncio.to_thread(
            predict_user, username, payload.gesture_number, model, feature_set,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/eval/leaderboard")
async def post_leaderboard(payload: EvalPayload) -> dict:
    try:
        return await asyncio.to_thread(eval_leaderboard, payload.feature_set, 5, payload.models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/eval/loou")
async def post_loou(payload: EvalPayload) -> dict:
    try:
        return await asyncio.to_thread(eval_loou, payload.feature_set, payload.models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/eval/confusion")
async def post_confusion(payload: ConfusionPayload) -> dict:
    try:
        return await asyncio.to_thread(eval_confusion, payload.feature_set, payload.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/users/{username}/gestures/{n}/data")
def get_saved_sensor_data(username: str, n: int, test: str = "") -> JSONResponse:
    path_dir = config.gesture_dir(username, n, test)
    if not path_dir.exists():
        raise HTTPException(status_code=404, detail="gesture data not found")
    out: dict[str, list[float]] = {}
    for i in range(1, 9):
        f = path_dir / f"A{i}.csv"
        if f.exists():
            with open(f) as fp:
                out[f"A{i}"] = [float(line.strip()) for line in fp if line.strip()]
        else:
            out[f"A{i}"] = []
    return JSONResponse(out)


@app.websocket("/ws/collect")
async def ws_collect(ws: WebSocket) -> None:
    await ws.accept()
    abort_event = asyncio.Event()
    pause_event = asyncio.Event()

    try:
        setup_raw = await ws.receive_text()
        setup = json.loads(setup_raw)
        username = setup["username"]
        gesture_number = int(setup["gesture_number"])
        mode = setup.get("mode", "train")
        variant = setup.get("variant", config.DEFAULT_VARIANT)
        if mode not in ("train", "test"):
            await ws.send_json({"event": "error", "detail": "mode must be train|test"})
            await ws.close()
            return
        if variant not in config.VARIANTS:
            variant = config.DEFAULT_VARIANT
    except Exception as e:
        await ws.send_json({"event": "error", "detail": f"bad setup: {e}"})
        await ws.close()
        return

    async def watch_client_messages() -> None:
        try:
            while True:
                msg = await ws.receive_text()
                try:
                    parsed = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                action = parsed.get("action")
                if action in ("abort", "stop"):
                    abort_event.set()
                    return
                if action == "pause":
                    pause_event.set()
                elif action == "resume":
                    pause_event.clear()
        except WebSocketDisconnect:
            abort_event.set()

    watcher = asyncio.create_task(watch_client_messages())

    try:
        async for event in collect_gesture(
            username, gesture_number, mode, abort_event, pause_event, variant=variant,
        ):
            await ws.send_json(event)
            if event.get("event") in ("done", "error", "aborted"):
                break
    except WebSocketDisconnect:
        abort_event.set()
    except Exception as e:
        try:
            await ws.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        watcher.cancel()
        try:
            await ws.close()
        except Exception:
            pass


if config.GESTURES_IMG_DIR.exists():
    app.mount("/gestures", StaticFiles(directory=str(config.GESTURES_IMG_DIR)), name="gestures")

app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
