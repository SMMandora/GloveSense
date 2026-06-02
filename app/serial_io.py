import threading
from typing import Optional

import serial
import serial.tools.list_ports as list_ports

from . import config


class SerialClient:
    _instance: Optional["SerialClient"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "SerialClient":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def open(self) -> None:
        if self._ser is None:
            self._ser = serial.Serial(
                port=config.SERIAL_PORT,
                baudrate=config.SERIAL_BAUD,
                timeout=config.SERIAL_TIMEOUT,
            )
        elif not self._ser.is_open:
            self._ser.open()
        else:
            self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def trigger_and_read_int(self) -> Optional[int]:
        if self._ser is None or not self._ser.is_open:
            return None
        self._ser.write(b"g")
        data = self._ser.readline().strip()
        try:
            return int(data.decode("utf-8", errors="ignore"))
        except ValueError:
            return None


def list_serial_ports() -> list[dict]:
    """Every serial port Windows currently sees."""
    out = []
    for p in list_ports.comports():
        out.append({
            "device": p.device,
            "name": p.name,
            "description": p.description,
            "manufacturer": p.manufacturer,
            "vid": p.vid,
            "pid": p.pid,
            "hwid": p.hwid,
        })
    return out


def glove_status() -> dict:
    """Detect whether the configured glove serial port is currently attached.

    Uses pyserial's port enumeration (cheap, doesn't open the port) so it is
    safe to call concurrently with an active collection. The 'busy' flag
    reflects whether a collection has the SerialClient lock right now.
    """
    ports = list_serial_ports()
    configured = config.SERIAL_PORT
    match = next((p for p in ports if p["device"].upper() == configured.upper()), None)
    client = SerialClient.get()
    locked = client.lock.locked()
    return {
        "port": configured,
        "baud": config.SERIAL_BAUD,
        "connected": match is not None,
        "device": match,
        "busy": locked,
        "ports_seen": [p["device"] for p in ports],
    }


def probe() -> None:
    """Smoke test: open the port, trigger a sample pair, print raw values."""
    client = SerialClient.get()
    client.open()
    print(f"Opened {config.SERIAL_PORT} @ {config.SERIAL_BAUD} baud")
    for _ in range(8):
        sensor = client.trigger_and_read_int()
        value = client.trigger_and_read_int()
        print(f"sensor={sensor} value={value}")
    client.close()


if __name__ == "__main__":
    print(glove_status())
