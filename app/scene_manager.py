"""场景管理模块。"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import inspect
from typing import Any

from app.quanser_shim import install_qcar2_shim


@dataclass
class SceneContext:
    scene_name: str
    obstacle_layout: str
    initial_pose: tuple[float, float, float]
    launched: bool
    setup_script: str | None
    handle: Any | None = None
    status: str = "prepared"


SCENE_SETUP_MAP = {
    "cityscape": Path("基于激光雷达的障碍物检测") / "qlabs_setup.py",
    "cityscape lite": Path("基于激光雷达的障碍物检测") / "qlabs_setup.py",
    "cityscapelite": Path("基于激光雷达的障碍物检测") / "qlabs_setup.py",
    "plane": Path("4_定位建图") / "4.1_雷达数据处理" / "qlabs_setup.py",
    "townscape": Path("3_行驶控制") / "3.2_车辆转向控制" / "qlabs_setup.py",
}

QLABS_EXE = Path(r"C:\Program Files\Quanser\Quanser Interactive Labs\Quanser Interactive Labs.exe")

SCENE_DEFAULT_POSE = {
    "cityscape": (0.0, 0.13024, -1.5707963267948966),
    "cityscape lite": (0.0, 0.13024, -1.5707963267948966),
    "cityscapelite": (0.0, 0.13024, -1.5707963267948966),
    "plane": (-0.031, 1.311, -1.5707963267948966),
    "townscape": (-0.74888, 0.83028, 0.0),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module_from_path(module_path: Path):
    spec = spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载场景脚本: {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qlabs_connectable() -> bool:
    try:
        from qvl.qlabs import QuanserInteractiveLabs
    except ImportError:
        return False

    qlabs = QuanserInteractiveLabs()
    try:
        ok = qlabs.open("localhost")
        if ok:
            qlabs.close()
        return bool(ok)
    except Exception:
        return False


def load_scene_module(scene_name: str, timeout: float = 30.0) -> bool:
    if not QLABS_EXE.exists():
        return False

    subprocess.Popen(
        [str(QLABS_EXE), "-loadmodule", scene_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if qlabs_connectable():
            time.sleep(2.0)
            return True
        time.sleep(1.0)
    return False


def prepare_scene(
    scene_name: str,
    obstacle_layout: str = "default",
    initial_pose: tuple[float, float, float] | None = None,
    launch: bool = False,
    enable_traffic: bool = False,
) -> SceneContext:
    """准备仿真场景。

    默认只返回场景上下文。`launch=True` 时尝试调用现有例程中的 qlabs_setup。
    """
    normalized_name = scene_name.strip().lower()
    if initial_pose is None:
        initial_pose = SCENE_DEFAULT_POSE.get(normalized_name, (0.0, 0.0, 0.0))
    setup_relpath = SCENE_SETUP_MAP.get(normalized_name)
    setup_abspath = _repo_root() / setup_relpath if setup_relpath else None
    handle = None
    launched = False

    if launch and setup_abspath and setup_abspath.exists():
        install_qcar2_shim()
        load_scene_module(scene_name)
        module = _load_module_from_path(setup_abspath)
        if hasattr(module, "setup"):
            setup_kwargs = {
                "initialPosition": [initial_pose[0], initial_pose[1], 0.0],
                "initialOrientation": [0.0, 0.0, initial_pose[2]],
            }
            signature = inspect.signature(module.setup)
            if "enableTraffic" in signature.parameters:
                setup_kwargs["enableTraffic"] = enable_traffic
            handle = module.setup(
                **setup_kwargs,
            )
            launched = True

    return SceneContext(
        scene_name=scene_name,
        obstacle_layout=obstacle_layout,
        initial_pose=initial_pose,
        launched=launched,
        setup_script=str(setup_abspath) if setup_abspath else None,
        handle=handle,
    )
