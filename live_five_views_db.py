# -*- coding: utf-8 -*-
# src/artifact_app/gui/live_five_views_widget.py
"""
LiveFiveViewsWidget V7 - חלון תצוגת 5 מבטים חיים עם QtInteractor

מבוסס על הגישה המוכחת של live_views_window.py:
  - camera_position ישיר (pos, focal, up)
  - _fit_parallel_scale לכל תא בנפרד
  - פריסה זהה לחלון הראשי (compute_matlab_5rects_normalized)

פריסה:
    TL  = Top view (top-left)
    MC  = Front/Dorsal (main)
    ML  = Left Profile (narrow, tall)
    MR  = Right Profile (narrow, tall)
    BR  = Bottom view (bottom-right)
    SCALE = Scale bar (bottom-center)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox,
)

try:
    from pyvistaqt import QtInteractor
    HAS_PYVISTAQT = True
except ImportError:
    HAS_PYVISTAQT = False

from artifact_app.viewer.view_matlab_style import (
    compute_matlab_5rects_normalized,
    NRect,
    _subplot33_positions_matlab_like,
)
from artifact_app.viewer.views_spec import get_views_spec
from artifact_app.viewer.main_view import compute_view_dirs_from_azel


# ============================================================
# Camera helpers - copied from live_views_window.py (proven)
# ============================================================

DIST_FACTOR = 1.5


def _compute_mesh_center(mesh: pv.PolyData) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return np.array([(xmax + xmin) * 0.5,
                     (ymax + ymin) * 0.5,
                     (zmax + zmin) * 0.5], dtype=float)


def _sample_points(mesh: pv.PolyData, max_n: int = 60000) -> np.ndarray:
    pts = np.asarray(mesh.points, dtype=float)
    if pts.shape[0] <= max_n:
        return pts
    step = max(1, pts.shape[0] // max_n)
    return pts[::step].copy()


def _rotate_vec_around_axis(v: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    theta = np.deg2rad(degrees)
    v_par = np.dot(v, axis) * axis
    v_perp = v - v_par
    w = np.cross(axis, v_perp)
    return v_par + v_perp * np.cos(theta) + w * np.sin(theta)


def _camera_from_angles(
    mesh: pv.PolyData,
    az: float, el: float, roll: float,
) -> tuple:
    """
    Compute (pos, focal, up) for given view angles.
    Same as LiveViewsWindow._camera_from_angles.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.array([
        (xmax + xmin) * 0.5,
        (ymax + ymin) * 0.5,
        (zmax + zmin) * 0.5,
    ], dtype=float)
    span = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    diag = float(np.linalg.norm(span)) or 1.0

    right, up, fwd = compute_view_dirs_from_azel(float(az), float(el))

    fwd = fwd / (np.linalg.norm(fwd) or 1.0)
    up = up / (np.linalg.norm(up) or 1.0)

    if roll:
        up = _rotate_vec_around_axis(up, axis=fwd, degrees=float(roll))
        right = np.cross(up, fwd)
        right = right / (np.linalg.norm(right) or 1.0)
        up = np.cross(fwd, right)
        up = up / (np.linalg.norm(up) or 1.0)

    dist = DIST_FACTOR * diag
    pos = center - dist * fwd

    return tuple(pos.tolist()), tuple(center.tolist()), tuple(up.tolist())


def _frame_from_base(base_angles: dict) -> np.ndarray:
    """
    Build orthonormal frame [right, up, fwd] from angles.
    Same as LiveViewsWindow._frame_from_base.
    """
    az = float(base_angles.get("az", 0.0))
    el = float(base_angles.get("el", 0.0))
    roll = float(base_angles.get("roll", 0.0))

    right, up, fwd = compute_view_dirs_from_azel(az, el)

    fwd = fwd / (np.linalg.norm(fwd) or 1.0)
    up = up / (np.linalg.norm(up) or 1.0)

    if roll:
        up = _rotate_vec_around_axis(up, axis=fwd, degrees=roll)

    right = np.cross(up, fwd)
    right = right / (np.linalg.norm(right) or 1.0)
    up = np.cross(fwd, right)
    up = up / (np.linalg.norm(up) or 1.0)

    return np.column_stack([right, up, fwd])


def _fit_parallel_scale(
    pts_sample: np.ndarray,
    mesh_center: np.ndarray,
    *,
    right: np.ndarray,
    up: np.ndarray,
    window_size: tuple[int, int],
    margin: float = 1.02,
) -> float:
    """
    Compute parallel_scale that makes mesh fill the view.
    Same as LiveViewsWindow._fit_parallel_scale.
    """
    w, h = window_size
    w = max(1, int(w))
    h = max(1, int(h))
    aspect = w / h

    right = right / (np.linalg.norm(right) + 1e-12)
    up = up / (np.linalg.norm(up) + 1e-12)

    d = pts_sample - mesh_center[None, :]
    r = d @ right
    u = d @ up

    h_span = float(r.max() - r.min())
    v_span = float(u.max() - u.min())

    base_ps = 0.5 * max(v_span, h_span / max(aspect, 1e-6), 1e-6)
    return float(base_ps * float(margin))


def _get_parallel_scale(p) -> float:
    try:
        return float(p.camera.GetParallelScale())
    except Exception:
        try:
            return float(p.renderer.GetActiveCamera().GetParallelScale())
        except Exception:
            return 0.0


def _set_parallel_scale(p, val: float) -> None:
    try:
        p.camera.SetParallelScale(float(val))
    except Exception:
        try:
            p.renderer.GetActiveCamera().SetParallelScale(float(val))
        except Exception:
            pass


# ============================================================
# Widget
# ============================================================

class LiveFiveViewsWidget(QWidget):
    """
    Widget with 5 live PyVista plotters arranged identically
    to MatlabFiveViewsCanvas.

    Camera setup uses the same proven approach as live_views_window.py.
    """

    VIEW_KEYS = ("TL", "ML", "MC", "MR", "BR")

    def __init__(
        self,
        mesh: pv.PolyData = None,
        parent: QWidget = None,
        locked: bool = True,
    ):
        super().__init__(parent)

        self._mesh: Optional[pv.PolyData] = None
        self._locked = locked
        self._views_spec = get_views_spec()

        # Bounding box for layout
        self._bbox_dx = 1.0
        self._bbox_dy = 1.0
        self._bbox_dz = 1.0

        # Cache for fit calculations
        self._mesh_center: Optional[np.ndarray] = None
        self._pts_sample: Optional[np.ndarray] = None

        # View angles (from views_spec)
        self._base_angles: Dict[str, Dict[str, float]] = {}

        # Plotter references
        self._plotters: Dict[str, QtInteractor] = {}
        self._mesh_actors: Dict[str, Any] = {}
        self._overlay_actors: Dict[str, List[Any]] = {}

        # Scale label
        self.lbl_scale: Optional[QLabel] = None

        # State
        self._plotters_ready = False

        self._build_ui()

        if mesh is not None:
            QTimer.singleShot(300, lambda: self.set_mesh(mesh))

    def _build_ui(self):
        """Create plotters as direct children (manual positioning)."""
        self.setStyleSheet("background: #DCE1E8;")

        if not HAS_PYVISTAQT:
            lbl = QLabel("pyvistaqt is required for live views", self)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setGeometry(0, 0, 400, 30)
            return

        for key in self.VIEW_KEYS:
            try:
                plotter = QtInteractor(self, auto_update=False)
                plotter.set_background("#B8C2CC")

                if self._locked:
                    try:
                        plotter.disable()
                    except Exception:
                        pass

                self._plotters[key] = plotter
                self._overlay_actors[key] = []
            except Exception as e:
                print(f"[LiveFiveViewsWidget] Failed to create plotter for {key}: {e}")

        # Scale label
        self.lbl_scale = QLabel("", self)
        self.lbl_scale.setAlignment(Qt.AlignCenter)
        self.lbl_scale.setStyleSheet(
            "background: #DCE1E8; border: 1px solid #ccc; font-size: 11px;"
        )

        self._plotters_ready = True

    # ----------------------------------------------------------------
    # Layout - identical to MatlabFiveViewsCanvas._layout_matlab_like()
    # ----------------------------------------------------------------
    def set_bbox(self, dx: float, dy: float, dz: float):
        self._bbox_dx = max(0.001, float(dx))
        self._bbox_dy = max(0.001, float(dy))
        self._bbox_dz = max(0.001, float(dz))
        self._do_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._do_layout()
        # Re-fit cameras after resize
        if self._mesh is not None:
            QTimer.singleShot(50, self._refit_all_cameras)

    def _do_layout(self):
        """Position plotters using compute_matlab_5rects_normalized."""
        W = self.width()
        H = self.height()
        if W < 50 or H < 50:
            return

        pad = 0
        inner_w = max(1, W - 2 * pad)
        inner_h = max(1, H - 2 * pad)

        dx = self._bbox_dx
        dy = self._bbox_dy
        dz = self._bbox_dz

        def to_qrect(nr: NRect) -> QRect:
            x = pad + int(round(nr.x * inner_w))
            w = int(round(nr.w * inner_w))
            y_top = 1.0 - (nr.y + nr.h)
            y = pad + int(round(y_top * inner_h))
            h = int(round(nr.h * inner_h))
            return QRect(x, y, max(1, w), max(1, h))

        if dx > 0 and dy > 0 and dz > 0:
            nrects = compute_matlab_5rects_normalized(dx, dy, dz)
            rects = {k: to_qrect(v) for k, v in nrects.items()}

            # Scale label position
            mid = nrects["ML"]
            bottom_key = min(("TL", "BR"), key=lambda k: float(nrects[k].y))
            bottom = nrects[bottom_key]
            rects["SCALE"] = to_qrect(NRect(mid.x, bottom.y, mid.w, bottom.h))
        else:
            pos = _subplot33_positions_matlab_like()
            rects = {}
            mapping = {9: "TL", 5: "ML", 4: "MC", 6: "MR", 1: "BR", 8: "SCALE"}
            for idx, key in mapping.items():
                if idx in pos:
                    rects[key] = to_qrect(pos[idx])

        for key in self.VIEW_KEYS:
            rect = rects.get(key)
            plotter = self._plotters.get(key)
            if rect is not None and plotter is not None:
                plotter.setGeometry(rect)

        if self.lbl_scale is not None and "SCALE" in rects:
            self.lbl_scale.setGeometry(rects["SCALE"])
            self.lbl_scale.raise_()
            self.lbl_scale.show()

    # ----------------------------------------------------------------
    # Mesh management - using live_views_window.py approach
    # ----------------------------------------------------------------
    def set_mesh(self, mesh: pv.PolyData):
        """Load mesh into all plotters with clean, high-res rendering."""
        if mesh is None or not self._plotters_ready:
            return

        # 1. חישוב נורמלים - קריטי להחזרת אור נכונה (מונע שטחיות)
        if mesh.n_points > 0:
            mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

        self._mesh = mesh
        self._mesh_center = _compute_mesh_center(mesh)
        self._pts_sample = _sample_points(mesh, max_n=60000)

        # Update bounding box for layout
        bounds = mesh.bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        self.set_bbox(dx, dy, dz)

        views = self._views_spec

        for key in self.VIEW_KEYS:
            plotter = self._plotters.get(key)
            if plotter is None:
                continue

            v = views.get(key, {}) if isinstance(views, dict) else {}
            if isinstance(v, dict):
                az = float(v.get("az", 0.0))
                el = float(v.get("el", 0.0))
                roll = float(v.get("roll", 0.0))
            else:
                az, el = float(v[0]), float(v[1])
                roll = float(v[2]) if len(v) > 2 else 0.0

            self._base_angles[key] = {"az": az, "el": el, "roll": roll}

            try:
                plotter.clear()

                # --- תיקון: הגדרות רינדור למראה נקי ועמוק ---

                # 1. החלקת קצוות (High Res look)
                plotter.enable_anti_aliasing()

                # 2. הוספת סט תאורה (פותר את בעיית הצללית השחורה!)
                plotter.enable_lightkit()

                plotter.set_background("#B8C2CC")

                # 3. הוספת המודל בצורה פשוטה (כמו בפונקציה הטובה)
                plotter.add_mesh(
                    self._mesh,
                    color="silver",
                    smooth_shading=True,  # מחליק את הפוליגונים
                    show_edges=False,
                    lighting=True  # מוודא שמגיב לאור
                )

                # Camera Setup
                pos, foc, up = _camera_from_angles(self._mesh, az, el, roll)
                plotter.camera_position = (pos, foc, up)

                try:
                    plotter.enable_parallel_projection()
                except Exception:
                    pass

                # Fit
                F_base = _frame_from_base(self._base_angles[key])
                fit_ps = _fit_parallel_scale(
                    self._pts_sample,
                    self._mesh_center,
                    right=F_base[:, 0],
                    up=F_base[:, 1],
                    window_size=(max(1, plotter.width()), max(1, plotter.height())),
                    margin=1.02,
                )
                _set_parallel_scale(plotter, fit_ps)

                if self._locked:
                    try:
                        plotter.disable()
                    except Exception:
                        pass

                plotter.render()

            except Exception as e:
                print(f"[LiveFiveViewsWidget] Error setting up {key}: {e}")

        self._update_scale_bar()
    def _refit_all_cameras(self):
        """Re-fit cameras after resize."""
        if self._mesh is None or self._pts_sample is None:
            return

        for key in self.VIEW_KEYS:
            plotter = self._plotters.get(key)
            base = self._base_angles.get(key)
            if plotter is None or base is None:
                continue

            try:
                F_base = _frame_from_base(base)
                fit_ps = _fit_parallel_scale(
                    self._pts_sample,
                    self._mesh_center,
                    right=F_base[:, 0],
                    up=F_base[:, 1],
                    window_size=(max(1, plotter.width()), max(1, plotter.height())),
                    margin=1.02,
                )
                _set_parallel_scale(plotter, fit_ps)
                plotter.render()
            except Exception:
                pass

        self._update_scale_bar()

    def _update_scale_bar(self):
        """Update scale bar label."""
        if self.lbl_scale is None or self._mesh is None:
            return

        try:
            from artifact_app.viewer.scale_bar import make_scale_pixmap

            mc_pl = self._plotters.get("MC")
            if mc_pl is None:
                return

            ps = _get_parallel_scale(mc_pl)
            w_win = max(1, mc_pl.width())
            h_win = max(1, mc_pl.height())

            ref_meta = {
                "parallel_scale": ps,
                "window_size": (w_win, h_win),
            }

            w_s = max(1, self.lbl_scale.width())
            h_s = max(1, self.lbl_scale.height())

            pix = make_scale_pixmap(
                w_s, h_s, ref_meta,
                target_frac=0.80,
                bg=Qt.white,
            )
            self.lbl_scale.setPixmap(pix)

        except Exception as e:
            self.lbl_scale.setText(f"Scale: {e}")

    # ----------------------------------------------------------------
    # Overlay API
    # ----------------------------------------------------------------
    def add_line(
        self,
        view_key: str,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        color: str = "red",
        width: float = 2.0,
    ) -> Optional[Any]:
        plotter = self._plotters.get(view_key)
        if plotter is None:
            return None

        line = pv.Line(start, end)
        actor = plotter.add_mesh(
            line, color=color, line_width=width,
            pickable=False, reset_camera=False,
        )
        self._overlay_actors[view_key].append(actor)
        plotter.render()
        return actor

    def add_polyline(
        self,
        view_key: str,
        points: np.ndarray,
        color: str = "red",
        width: float = 2.0,
        closed: bool = False,
    ) -> Optional[Any]:
        plotter = self._plotters.get(view_key)
        if plotter is None or len(points) < 2:
            return None

        if closed:
            points = np.vstack([points, points[0]])

        polyline = pv.lines_from_points(points, close=False)
        actor = plotter.add_mesh(
            polyline, color=color, line_width=width,
            pickable=False, reset_camera=False,
        )
        self._overlay_actors[view_key].append(actor)
        plotter.render()
        return actor

    def clear_overlays(self, view_key: str = None):
        keys = [view_key] if view_key else list(self._overlay_actors.keys())
        for key in keys:
            plotter = self._plotters.get(key)
            if plotter is None:
                continue
            for actor in self._overlay_actors.get(key, []):
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
            self._overlay_actors[key] = []
            plotter.render()

    def get_plotter(self, view_key: str) -> Optional[Any]:
        return self._plotters.get(view_key)

    # ----------------------------------------------------------------
    # Screenshot
    # ----------------------------------------------------------------
    def screenshot(self, path: str = None) -> Optional[np.ndarray]:
        pixmap = self.grab()
        if path:
            pixmap.save(path)
            return None
        else:
            img = pixmap.toImage()
            w, h = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(h * w * 4)
            arr = np.array(ptr).reshape(h, w, 4)
            return arr[:, :, :3]

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    def close(self):
        for plotter in self._plotters.values():
            try:
                plotter.close()
            except Exception:
                pass
        self._plotters.clear()
        super().close()


# ============================================================
# Standalone Window
# ============================================================

class LiveFiveViewsWindow(QWidget):

    def __init__(
        self,
        mesh: pv.PolyData,
        title: str = "Export Views",
        parent: QWidget = None,
        filename: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Export Views: {filename}" if filename else title)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1000, 850)

        self._filename = filename
        self._mesh = mesh

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        lbl = QLabel(filename or "Object")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #444; "
            "background-color: #A7B4C6; padding: 4px;"
        )
        layout.addWidget(lbl)

        self.live_views = LiveFiveViewsWidget(mesh, parent=self, locked=True)
        layout.addWidget(self.live_views, 1)

        btn_frame = QWidget()
        btn_frame.setStyleSheet("background: white;")
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(8, 8, 8, 8)
        btn_layout.addStretch()

        btn_save_jpg = QPushButton("💾 Save JPG")
        btn_save_jpg.clicked.connect(lambda: self._save("jpg"))
        btn_layout.addWidget(btn_save_jpg)

        btn_save_png = QPushButton("💾 Save PNG")
        btn_save_png.clicked.connect(lambda: self._save("png"))
        btn_layout.addWidget(btn_save_png)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addWidget(btn_frame)

    def _save(self, fmt: str):
        name = self._filename
        for ext in (".wrl", ".ply", ".obj", ".stl"):
            name = name.replace(ext, "")
        name = name or "object"
        default = f"{name} - full patch.{fmt}"

        path, _ = QFileDialog.getSaveFileName(
            self, f"Save as {fmt.upper()}", default,
            f"{fmt.upper()} (*.{fmt})"
        )
        if path:
            self.live_views.screenshot(path)
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")

    def closeEvent(self, event):
        self.live_views.close()
        super().closeEvent(event)