"""Quanser 运行时兼容补丁。"""

from __future__ import annotations

import builtins
import io
import json
import sys
import types
from pathlib import Path


QCAR2_CONFIG = {
    "cartype": 2,
    "carname": "qcar2",
    "lidarurl": "serial-cpu://localhost:1?baud='256000',word='8',parity='none',stop='1',flow='none',dsr='on'",
    "WHEEL_RADIUS": 0.033,
    "WHEEL_BASE": 0.256,
    "PIN_TO_SPUR_RATIO": (13.0 * 19.0) / (70.0 * 37.0),
    "WRITE_PWM_CHANNELS": [-1],
    "WRITE_OTHER_CHANNELS": [1000, 11000],
    "WRITE_DIGITAL_CHANNELS": [17, 18, 25, 26, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24],
    "writePWMBuffer": 1,
    "writeDigitalBuffer": 16,
    "writeOtherBuffer": 2,
    "READ_ANALOG_CHANNELS": [4, 2],
    "READ_ENCODER_CHANNELS": [0],
    "READ_OTHER_CHANNELS": [3000, 3001, 3002, 4000, 4001, 4002, 14000],
    "readAnalogBuffer": 2,
    "readEncoderBuffer": 1,
    "readOtherBuffer": 7,
    "csiRight": 0,
    "csiBack": 1,
    "csiLeft": 3,
    "csiFront": 2,
    "lidarToGps": "qcar2LidarToGPS.rt-linux_qcar2",
    "captureScan": "qcar2CaptureScan.rt-linux_qcar2",
}


_INSTALLED = False


def install_qcar2_shim() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    fake_module = types.ModuleType("pal.products.qcar_config")

    class FakeQCarCheck:
        def __init__(self, *args, **kwargs):
            pass

    fake_module.QCar_check = FakeQCarCheck
    sys.modules["pal.products.qcar_config"] = fake_module

    real_open = builtins.open
    config_path = str(
        Path.home() / "Documents" / "Quanser" / "0_libraries" / "python" / "pal" / "products" / "qcar_config.json"
    ).replace("/", "\\").lower()
    config_text = json.dumps(QCAR2_CONFIG)

    def patched_open(file, mode="r", *args, **kwargs):
        try:
            candidate = str(Path(file)).replace("/", "\\").lower()
        except Exception:
            candidate = ""
        if candidate == config_path and "r" in mode and "w" not in mode and "a" not in mode:
            return io.StringIO(config_text)
        return real_open(file, mode, *args, **kwargs)

    builtins.open = patched_open
    _INSTALLED = True
