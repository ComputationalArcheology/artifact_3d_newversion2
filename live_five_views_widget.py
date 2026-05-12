# -*- coding: utf-8 -*-
# src/artifact_app/gui/live_five_views_widget.py
"""
LiveFiveViewsWidget V14 — שימוש בלוגיקת Layout שכבר עובדת

הגישה:
  1. ה-layout של התצוגה הראשית (MatlabFiveViewsCanvas) כבר עובד מושלם
     עם compute_matlab_5rects_normalized(dx, dy, dz).
  2. הפונקציה הזו מקבלת bbox dims ומחשבת rxy, ryz, rzx — בדיוק כמו במטלב.
  3. אנחנו פשוט קוראים לה גם כאן ומיישמים את ה-NRects על הפלוטרים.

  השינוי היחיד: במקום dx/dy/dz מ-bbox, אנחנו מחשבים "effective" dx/dy/dz
  מההטלות בפועל — כדי שזה יעבוד גם אם ה-mesh לא axis-aligned.
  (בפועל, אחרי align_mesh, ה-bbox מספיק טוב.)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt, QTimer, QRect, QEvent, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox,
)

try:
    from pyvistaqt import QtInteractor

    HAS_PYVISTAQT = True
except ImportError:
    HAS_PYVISTAQT = False

# *** שימוש ב-layout שכבר עובד ***
from artifact_app.viewer.view_matlab_style import (
    compute_matlab_5rects_normalized,
    NRect,
    _subplot33_positions_matlab_like,
)
from artifact_app.viewer.views_spec import get_views_spec
from artifact_app.viewer.main_view import compute_view_dirs_from_azel

# ============================================================
# Camera helpers
# ============================================================
DIST_FACTOR = 2.0


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


def _camera_from_angles(mesh: pv.PolyData, az: float, el: float, roll: float) -> tuple:
    center = _compute_mesh_center(mesh)
    bounds = mesh.bounds
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    right, up, fwd = compute_view_dirs_from_azel(float(az), float(el))
    pos = center - fwd * (diag * DIST_FACTOR)
    if roll:
        up = _rotate_vec_around_axis(up, axis=fwd, degrees=float(roll))
    return tuple(pos.tolist()), tuple(center.tolist()), tuple(up.tolist())


def _frame_from_base(base_angles: dict) -> np.ndarray:
    """Return [right, up, fwd] column matrix."""
    az = float(base_angles.get("az", 0.0))
    el = float(base_angles.get("el", 0.0))
    roll = float(base_angles.get("roll", 0.0))
    right, up, fwd = compute_view_dirs_from_azel(az, el)
    if roll:
        up = _rotate_vec_around_axis(up, axis=fwd, degrees=roll)
    right = np.cross(up, fwd)
    right = right / (np.linalg.norm(right) or 1.0)
    up = np.cross(fwd, right)
    up = up / (np.linalg.norm(up) or 1.0)
    return np.column_stack([right, up, fwd])


def _projection_spans(pts: np.ndarray, center: np.ndarray,
                      right: np.ndarray, up: np.ndarray) -> Tuple[float, float]:
    """Project mesh points, return (h_span, v_span) in world units."""
    d = pts - center[None, :]
    r_proj = d @ right
    u_proj = d @ up
    h_span = float(r_proj.max() - r_proj.min())
    v_span = float(u_proj.max() - u_proj.min())
    return max(h_span, 1e-9), max(v_span, 1e-9)


def _set_parallel_scale(p, val: float) -> None:
    try:
        p.camera.SetParallelScale(float(val))
    except Exception:
        pass


# ============================================================
# Widget
# ============================================================

class LiveFiveViewsWidget(QWidget):
    VIEW_KEYS = ("TL", "ML", "MC", "MR", "BR")
    pointClicked = Signal(str, float, float, float)

    def __init__(self, mesh: pv.PolyData = None, parent: QWidget = None, locked: bool = True):
        super().__init__(parent)
        self._mesh: Optional[pv.PolyData] = None
        self._locked = locked
        self._views_spec = get_views_spec()

        # bbox dims for layout — same as MatlabFiveViewsCanvas
        self._bbox_dx = 0.0
        self._bbox_dy = 0.0
        self._bbox_dz = 0.0

        self._mesh_center: Optional[np.ndarray] = None
        self._pts_sample: Optional[np.ndarray] = None
        self._base_angles: Dict[str, Dict[str, float]] = {}

        self._plotters: Dict[str, QtInteractor] = {}
        self._overlay_actors: Dict[str, List[Any]] = {}
        self._coord_labels: Dict[str, QLabel] = {}
        self._plotter_to_key: Dict[QtInteractor, str] = {}

        self.lbl_scale: Optional[QLabel] = None
        self._plotters_ready = False

        self._build_ui()
        if mesh is not None:
            QTimer.singleShot(300, lambda: self.set_mesh(mesh))

    def _build_ui(self):
        bg_color = "#F0F0F0"
        self.setStyleSheet(f"background: {bg_color};")

        if not HAS_PYVISTAQT:
            lbl = QLabel("Error: pyvistaqt is required", self)
            return

        for key in self.VIEW_KEYS:
            try:
                plotter = QtInteractor(self, auto_update=False)
                plotter.set_background(bg_color)

                if self._locked:
                    try:
                        plotter.disable()
                    except:
                        pass

                self._plotters[key] = plotter
                self._overlay_actors[key] = []
                self._plotter_to_key[plotter] = key

                plotter.setMouseTracking(True)
                plotter.installEventFilter(self)

                coord_label = QLabel("", self)
                coord_label.setStyleSheet(
                    "background: rgba(0,0,0,0.6); color: #00FF00; padding: 2px; border-radius: 3px;")
                coord_label.hide()
                self._coord_labels[key] = coord_label

            except Exception as e:
                print(f"Error creating plotter {key}: {e}")

        self.lbl_scale = QLabel("", self)
        self.lbl_scale.setAlignment(Qt.AlignCenter)
        self.lbl_scale.setStyleSheet("color: #333; font-weight: bold; font-size: 11px;")

        self._plotters_ready = True

    # ----------------------------------------------------------------
    # Layout — uses the SAME function as MatlabFiveViewsCanvas
    # ----------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._do_layout()
        if self._mesh is not None:
            QTimer.singleShot(50, self._refit_all_cameras)

    def _do_layout(self):
        W = self.width()
        H = self.height()
        if W < 50 or H < 50: return

        inner_w = W
        inner_h = H

        def to_qrect(nr: NRect) -> QRect:
            x = int(round(nr.x * inner_w))
            w = int(round(nr.w * inner_w))
            y_top = 1.0 - (nr.y + nr.h)
            y = int(round(y_top * inner_h))
            h = int(round(nr.h * inner_h))
            return QRect(x, y, max(1, w), max(1, h))

        rects = {}
        if self._bbox_dx > 0 and self._bbox_dy > 0 and self._bbox_dz > 0:
            # *** EXACT same call as MatlabFiveViewsCanvas ***
            nrects = compute_matlab_5rects_normalized(self._bbox_dx, self._bbox_dy, self._bbox_dz)
            rects = {k: to_qrect(v) for k, v in nrects.items()}

            # Scale label position
            mid = nrects["ML"]
            bottom_key = min(("TL", "BR"), key=lambda k: float(nrects[k].y))
            bottom = nrects[bottom_key]
            rects["SCALE"] = to_qrect(NRect(mid.x, bottom.y, mid.w, bottom.h))
        else:
            # Fallback: equal grid before mesh is loaded
            pos = _subplot33_positions_matlab_like()
            mapping = {9: "TL", 5: "ML", 4: "MC", 6: "MR", 1: "BR"}
            for idx, key in mapping.items():
                rects[key] = to_qrect(pos[idx])
            rects["SCALE"] = to_qrect(pos[8])

        for key in self.VIEW_KEYS:
            if key in rects and key in self._plotters:
                self._plotters[key].setGeometry(rects[key])
                self._plotters[key].show()

        if self.lbl_scale and "SCALE" in rects:
            self.lbl_scale.setGeometry(rects["SCALE"])
            self.lbl_scale.raise_()
            self.lbl_scale.show()

    # ----------------------------------------------------------------
    # Mesh & Rendering
    # ----------------------------------------------------------------
    def set_mesh(self, mesh: pv.PolyData):
        if mesh is None or not self._plotters_ready:
            return

        if mesh.n_points > 0:
            mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

        self._mesh = mesh
        self._mesh_center = _compute_mesh_center(mesh)
        self._pts_sample = _sample_points(mesh, max_n=60000)

        # Set bbox dims — SAME as what MainWindow does
        bounds = mesh.bounds
        self._bbox_dx = bounds[1] - bounds[0]
        self._bbox_dy = bounds[3] - bounds[2]
        self._bbox_dz = bounds[5] - bounds[4]

        # Trigger layout with correct bbox dims
        self._do_layout()

        # Precompute view angles
        for key in self.VIEW_KEYS:
            v_spec = self._views_spec.get(key, {})
            self._base_angles[key] = {
                "az": v_spec.get("az", 0.0),
                "el": v_spec.get("el", 0.0),
                "roll": v_spec.get("roll", 0.0),
            }

        # Add mesh to plotters and set cameras
        # Step A: compute world_per_px for each view, then take global max
        # (same logic as render_views_pixmaps_by_sizes with GLOBAL_RULER_MARGIN)
        view_data = []

        for key in self.VIEW_KEYS:
            plotter = self._plotters.get(key)
            if plotter is None: continue

            plotter.clear()
            plotter.enable_anti_aliasing()
            plotter.enable_lightkit()

            plotter.add_mesh(self._mesh, color="#AAAAAA", smooth_shading=True, show_edges=False)

            base = self._base_angles[key]
            cam_pos, foc, up = _camera_from_angles(self._mesh, base["az"], base["el"], base["roll"])
            plotter.camera_position = (cam_pos, foc, up)
            plotter.enable_parallel_projection()

            # Compute projection spans and world_per_px
            F = _frame_from_base(base)
            h_span, v_span = _projection_spans(self._pts_sample, self._mesh_center, right=F[:, 0], up=F[:, 1])

            pw = max(1, plotter.width())
            ph = max(1, plotter.height())

            # world_per_px: how many world units per pixel needed to fit the object exactly in this plotter
            if h_span / pw > v_span / ph:
                wpp = h_span / pw
            else:
                wpp = v_span / ph

            view_data.append({"key": key, "wpp": wpp})

        # Step B: global world_per_px (largest wins = all views same scale)
        if view_data:
            global_wpp = max(vd["wpp"] for vd in view_data)

            for key in self.VIEW_KEYS:
                plotter = self._plotters.get(key)
                if plotter is None: continue
                # ParallelScale = half the visible height in world units
                # visible_height = global_wpp * plotter_height_px
                ph = max(1, plotter.height())
                ps = global_wpp * ph / 2.0
                _set_parallel_scale(plotter, ps)
                try:
                    plotter.enable()
                    plotter.enable_trackball_style()
                except:
                    pass
                plotter.render()

        self._update_scale_bar()

    def _refit_all_cameras(self):
        """Called on resize — recalculate ParallelScale using global world_per_px."""
        if self._mesh is None or self._pts_sample is None: return

        view_data = []
        for key in self.VIEW_KEYS:
            plotter = self._plotters.get(key)
            base = self._base_angles.get(key)
            if plotter is None or base is None: continue

            F = _frame_from_base(base)
            h_span, v_span = _projection_spans(self._pts_sample, self._mesh_center, right=F[:, 0], up=F[:, 1])

            pw = max(1, plotter.width())
            ph = max(1, plotter.height())

            if h_span / pw > v_span / ph:
                wpp = h_span / pw
            else:
                wpp = v_span / ph

            view_data.append({"key": key, "wpp": wpp})

        if view_data:
            global_wpp = max(vd["wpp"] for vd in view_data)
            for key in self.VIEW_KEYS:
                plotter = self._plotters.get(key)
                if plotter is None: continue
                ph = max(1, plotter.height())
                ps = global_wpp * ph / 2.0
                _set_parallel_scale(plotter, ps)
                plotter.render()

        self._update_scale_bar()

    def _update_scale_bar(self):
        """פונקציה זו רוקנה בכוונה כדי להסתיר את טקסט ה-Scale מהמסך"""
        if self.lbl_scale is not None:
            self.lbl_scale.setText("")
    # ----------------------------------------------------------------
    # Events / Utils
    # ----------------------------------------------------------------
    def eventFilter(self, obj, event):
        if not isinstance(obj, QtInteractor): return super().eventFilter(obj, event)
        key = self._plotter_to_key.get(obj)
        if key is None: return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseMove:
            self._handle_mouse_move(obj, key, event)
            return False
        elif event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._handle_mouse_click(obj, key, event)
            return False
        elif event.type() == QEvent.Leave:
            if key in self._coord_labels: self._coord_labels[key].hide()
            return False
        return super().eventFilter(obj, event)

    def _handle_mouse_move(self, plotter, key, event):
        if self._mesh is None: return
        try:
            try:
                x, y = int(event.position().x()), int(event.position().y())
            except AttributeError:
                x, y = event.x(), event.y()
            point = self._pick_point(plotter, x, y)
            lbl = self._coord_labels.get(key)
            if lbl and point:
                px, py, pz = point
                lbl.setText(f"({px:.2f}, {py:.2f}, {pz:.2f})")
                g_pos = plotter.mapTo(self, event.position().toPoint() if hasattr(event, 'position') else event.pos())
                lbl.move(g_pos.x() + 15, g_pos.y() - 15)
                lbl.adjustSize()
                lbl.show()
                lbl.raise_()
            elif lbl:
                lbl.hide()
        except:
            pass

    def _handle_mouse_click(self, plotter, key, event):
        try:
            try:
                x, y = int(event.position().x()), int(event.position().y())
            except AttributeError:
                x, y = event.x(), event.y()
            point = self._pick_point(plotter, x, y)
            if point: self.pointClicked.emit(key, *point)
        except:
            pass

    def _pick_point(self, plotter, x, y):
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker
            picker = vtkCellPicker()
            picker.SetTolerance(0.005)
            if picker.Pick(x, plotter.height() - y, 0, plotter.renderer):
                return tuple(map(float, picker.GetPickPosition()))
        except:
            pass
        return None

    def screenshot(self, path=None):
        for l in self._coord_labels.values(): l.hide()
        pix = self.grab()
        if path:
            pix.save(path)
            return None
        img = pix.toImage()
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(h * w * 4)
        return np.array(ptr).reshape(h, w, 4)[:, :, :3]

    def close(self):
        for p in self._plotters.values():
            try:
                p.close()
            except:
                pass
        self._plotters.clear()
        super().close()


# ============================================================
# Standalone Window - (פתרון שגיאת ה-Import)
# ============================================================
class LiveFiveViewsWindow(QWidget):
    def __init__(self, mesh: pv.PolyData, title="Export Views", parent=None, filename=""):
        super().__init__(parent)
        self.setWindowTitle(f"Export Views: {filename}" if filename else title)
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1000, 850)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if filename:
            lbl = QLabel(filename)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; background: #ddd; padding: 4px;")
            layout.addWidget(lbl)

        self.live_views = LiveFiveViewsWidget(mesh, parent=self, locked=False)
        layout.addWidget(self.live_views, 1)

        btn_layout = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def closeEvent(self, event):
        self.live_views.close()
        super().closeEvent(event)