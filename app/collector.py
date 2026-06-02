import asyncio
from typing import AsyncIterator, Optional

import numpy as np

from . import config
from .features import (
    compute_energy_features,
    normalize_energy_rows,
    save_energy_csv,
    save_raw_backup,
)
from .serial_io import SerialClient
from .signal_processing import process_signal
from .storage import create_path_folder


REST_DIVISIONS = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
BEEP_DIVISIONS = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}


async def collect_gesture(
    username: str,
    gesture_number: int,
    mode: str,
    abort_event: Optional[asyncio.Event] = None,
    pause_event: Optional[asyncio.Event] = None,
    variant: str = config.DEFAULT_VARIANT,
) -> AsyncIterator[dict]:
    """Stream sample / phase / done events while collecting from the glove.

    For mode == "train", `variant` selects the collection profile (v10 / v5)
    which controls target sample count, data-stream window, and J-file count.
    Test collection is identical regardless of variant.
    """
    test_suffix = "" if mode == "train" else "Test"
    if mode == "train":
        v = config.VARIANTS.get(variant) or config.VARIANTS[config.DEFAULT_VARIANT]
        target = v["train_samples"]
        j_files = v["train_windows"]
        data_stream = v["data_stream"]
    else:
        target = config.TEST_SAMPLES_PER_SENSOR
        j_files = config.TEST_WINDOWS
        data_stream = config.TEST_DATA_STREAM

    create_path_folder(username, gesture_number, test_suffix)
    gesture_letter = config.GESTURE_MAPPING.get(gesture_number, str(gesture_number))
    yield {
        "event": "started",
        "username": username,
        "gesture_number": gesture_number,
        "gesture_letter": gesture_letter,
        "mode": mode,
        "variant": variant if mode == "train" else None,
        "target_samples": target,
        "j_files": j_files,
    }
    yield {"event": "phase", "phase": "rest", "image": config.REST_IMAGE_ID}

    client = SerialClient.get()
    if not client.lock.acquire(blocking=False):
        yield {"event": "error", "detail": "serial port busy"}
        return

    try:
        client.open()
        await asyncio.sleep(1)

        sensors: list[list[int]] = [[] for _ in range(8)]
        last_phase: Optional[str] = None

        while True:
            if abort_event is not None and abort_event.is_set():
                yield {"event": "aborted"}
                return

            if pause_event is not None and pause_event.is_set():
                yield {"event": "paused"}
                while pause_event.is_set():
                    if abort_event is not None and abort_event.is_set():
                        yield {"event": "aborted"}
                        return
                    await asyncio.sleep(0.1)
                yield {"event": "resumed"}

            sensor_num = await asyncio.to_thread(client.trigger_and_read_int)
            if sensor_num is None:
                continue
            sensor_value = await asyncio.to_thread(client.trigger_and_read_int)
            if sensor_value is None:
                continue

            if 1 <= sensor_num <= 8:
                sensors[sensor_num - 1].append(sensor_value)
                yield {
                    "event": "sample",
                    "sensor": sensor_num,
                    "value": sensor_value,
                    "count": len(sensors[sensor_num - 1]),
                    "target": target,
                }
            else:
                continue

            if all(len(arr) % 5 == 0 for arr in sensors):
                division = len(sensors[0]) // 5
                if division in REST_DIVISIONS and last_phase != "rest":
                    last_phase = "rest"
                    yield {"event": "phase", "phase": "rest", "image": config.REST_IMAGE_ID}
                    await asyncio.sleep(2)
                elif division in BEEP_DIVISIONS and last_phase != "beep":
                    last_phase = "beep"
                    yield {"event": "phase", "phase": "beep", "image": int(gesture_number)}
                    await asyncio.sleep(2)

                min_length = min(len(arr) for arr in sensors)
                if min_length == target:
                    paths = _finalize_gesture(
                        username, gesture_number, gesture_letter, test_suffix,
                        sensors, mode, j_files, data_stream,
                    )
                    yield {"event": "done", "paths": paths}
                    return

            await asyncio.sleep(0)
    finally:
        client.close()
        client.lock.release()


def _finalize_gesture(
    username: str,
    gesture_number: int,
    gesture_letter: str,
    test_suffix: str,
    sensors: list[list[int]],
    mode: str,
    j_files: int,
    data_stream: int,
) -> dict:
    out_dir = config.gesture_dir(username, gesture_number, test_suffix)
    log_path = out_dir / "sensor_values.log"
    with open(log_path, "a") as log_file:
        parts = [str(arr[5:]) for arr in sensors]
        log_file.write("A1, A2, A3, A4, A5, A6, A7, A8: " + " ".join(parts) + "\n")

    A = []
    for arr in sensors:
        a = np.array(arr[5:], dtype=float)
        a -= np.mean(a[:5])
        A.append(a)

    VR: list[int] = []
    VM: list[int] = []
    for i in range(0, data_stream, 10):
        VR.extend(range(i, i + 5))
        VM.extend(range(i + 5, i + 10))
    VR_arr = np.array(VR)
    VM_arr = np.array(VM)

    A = [process_signal(a, VR_arr, VM_arr) for a in A]
    for a in A:
        a[VR_arr] = 0

    stacked = np.vstack(A)
    for i in VM_arr:
        max_val = np.max(np.abs(stacked[:, i]))
        if max_val != 0:
            stacked[:, i] = stacked[:, i] / max_val

    proc_log_path = out_dir / "sensor_values_after_processing.log"
    with open(proc_log_path, "a") as log_file:
        parts = [str(row) for row in stacked]
        log_file.write("A1, A2, A3, A4, A5, A6, A7, A8: " + " ".join(parts) + "\n")

    csv_paths = []
    for i in range(8):
        path = out_dir / f"A{i + 1}.csv"
        np.savetxt(path, stacked[i], delimiter=",")
        csv_paths.append(str(path))

    # Clear any stale J-files from a previous run/variant before saving the
    # fresh set, so a gesture folder always reflects the latest collection.
    for stale in out_dir.glob(f"J{gesture_number}.*.csv"):
        try:
            stale.unlink()
        except OSError:
            pass

    energies = compute_energy_features(stacked, VM_arr, j_files)
    normalized = normalize_energy_rows(energies)

    j_paths = []
    for k in range(j_files):
        col = normalized[:, k]
        j_path = save_energy_csv(username, gesture_number, test_suffix, k + 1, col.tolist())
        j_paths.append(j_path)

    raw_path = save_raw_backup(username, gesture_number, gesture_letter, sensors) if mode == "train" else None

    return {"a_csv": csv_paths, "j_csv": j_paths, "raw_csv": raw_path, "dir": str(out_dir)}
