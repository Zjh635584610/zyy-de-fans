"""结果显示模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisualizerConfig:
    enable_gui: bool = False
    title: str = "Obstacle Detection Live"
    fps: int = 20


class Visualizer:
    """统一管理点云、地图、障碍物框和状态显示。"""

    def __init__(self, config: VisualizerConfig | None = None) -> None:
        self.config = config or VisualizerConfig()
        self.last_frame = None
        self.gui_enabled = False
        self.scope = None
        self.pg = None
        self.MultiScope = None
        self.path_points: list[tuple[float, float]] = []
        self.box_items = []

        if self.config.enable_gui:
            self._setup_gui()

    def _setup_gui(self) -> None:
        try:
            import pyqtgraph as pg
            from pal.utilities.scope import MultiScope
        except Exception:
            self.gui_enabled = False
            return

        self.pg = pg
        self.MultiScope = MultiScope
        self.scope = MultiScope(rows=2, cols=2, title=self.config.title, fps=self.config.fps)

        self.scope.addXYAxis(row=0, col=0, xLabel="Angle [deg]", yLabel="Range [m]")
        self.scope.axes[0].attachImage()

        self.scope.addXYAxis(row=1, col=0, xLabel="Local X [m]", yLabel="Local Y [m]")
        self.scope.axes[1].attachImage()

        self.scope.addXYAxis(
            row=0,
            col=1,
            rowSpan=2,
            xLabel="World X [m]",
            yLabel="World Y [m]",
            xLim=(-4, 4),
            yLim=(-4, 8),
        )
        self.scope.axes[2].attachImage()

        self.scope.axes[0].images[0].rotation = 90
        self.scope.axes[0].images[0].levels = (0, 1)
        self.scope.axes[1].images[0].levels = (0, 1)
        self.scope.axes[2].images[0].levels = (0, 1)

        self.trajectory_item = pg.PlotDataItem(
            pen={"color": (85, 168, 104), "width": 2},
            name="Trajectory",
        )
        self.scope.axes[2].plot.addItem(self.trajectory_item)

        self.point_scatter = pg.ScatterPlotItem(
            size=2,
            brush=pg.mkBrush(255, 255, 255, 120),
            pen=pg.mkPen(None),
        )
        self.scope.axes[2].plot.addItem(self.point_scatter)

        self.center_scatter = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(196, 78, 82, 220),
            pen=pg.mkPen("w"),
        )
        self.scope.axes[2].plot.addItem(self.center_scatter)

        self.gui_enabled = True

    def _ensure_box_items(self, count: int) -> None:
        if not self.gui_enabled or self.pg is None or self.scope is None:
            return
        while len(self.box_items) < count:
            item = self.pg.PlotCurveItem(pen={"color": (246, 198, 68), "width": 2})
            self.scope.axes[2].plot.addItem(item)
            self.box_items.append(item)
        while len(self.box_items) > count:
            item = self.box_items.pop()
            self.scope.axes[2].plot.removeItem(item)

    def render(
        self,
        ego_state: dict,
        tracked_objects: list[dict],
        decision: dict,
        map_state: dict | None = None,
        point_cloud_world: np.ndarray | None = None,
    ) -> dict:
        self.last_frame = {
            "ego_state": ego_state,
            "tracked_objects": tracked_objects,
            "decision": decision,
            "map_state": map_state,
            "point_cloud_world": point_cloud_world,
        }

        if not self.gui_enabled or self.scope is None:
            return self.last_frame

        if map_state is not None:
            polar = np.asarray(map_state["polar_prob"], dtype=float)
            local = np.asarray(map_state["local_prob"], dtype=float)
            world = np.asarray(map_state["global_prob"], dtype=float)
            self.scope.axes[0].images[0].scale = (map_state["r_res"], -np.rad2deg(map_state["phi_res"]))
            self.scope.axes[0].images[0].offset = (0, 0)
            self.scope.axes[0].images[0].setImage(image=polar)

            local_width = map_state["local_prob"].shape[0] * map_state["cell_width"]
            self.scope.axes[1].images[0].scale = (map_state["cell_width"], -map_state["cell_width"])
            self.scope.axes[1].images[0].offset = (-local_width / 2, local_width / 2)
            self.scope.axes[1].images[0].setImage(image=local)

            self.scope.axes[2].images[0].scale = (map_state["cell_width"], -map_state["cell_width"])
            self.scope.axes[2].images[0].offset = (
                map_state["x_min"] / map_state["cell_width"],
                -map_state["y_max"] / map_state["cell_width"],
            )
            self.scope.axes[2].images[0].setImage(image=world)

        pose = ego_state["pose"]
        self.path_points.append((float(pose["x"]), float(pose["y"])))
        if self.path_points:
            path = np.array(self.path_points, dtype=float)
            self.trajectory_item.setData(path[:, 0], path[:, 1])

        if point_cloud_world is not None and len(point_cloud_world):
            points = np.asarray(point_cloud_world, dtype=float)
            self.point_scatter.setData(x=points[:, 0], y=points[:, 1])
        else:
            self.point_scatter.setData(x=[], y=[])

        centers = np.array([obj["center"][:2] for obj in tracked_objects], dtype=float) if tracked_objects else np.empty((0, 2))
        if centers.size:
            self.center_scatter.setData(x=centers[:, 0], y=centers[:, 1])
        else:
            self.center_scatter.setData(x=[], y=[])

        self._ensure_box_items(len(tracked_objects))
        for item, obj in zip(self.box_items, tracked_objects):
            x0, y0, _ = obj["bbox_min"]
            x1, y1, _ = obj["bbox_max"]
            item.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

        return self.last_frame

    def refresh(self) -> None:
        if self.gui_enabled and self.MultiScope is not None:
            self.MultiScope.refreshAll()
