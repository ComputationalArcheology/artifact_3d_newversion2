# -*- coding: utf-8 -*-
# src/artifact_app/gui/live_views_window.py
from __future__ import annotations

from artifact_app.viewer.main_view import compute_view_dirs_from_azel

from typing import Dict, Any
from datetime import datetime
import os

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QImage, QPainter, QAction
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout,
    QSplitter, QLabel, QSizePolicy,
    QFileDialog, QMessageBox,
    QToolButton, QMenu,
)

from artifact_app.gui.widgets import ArtifactButton, ArtifactToolButton


class SquarePlotterWidget(QWidget):
    """
    Wrapper שמכריח את ה-QtInteractor להיות ריבועי וממורכז בתוך התא.
    זה עוזר לשמור על גודל "אחיד" ויציב בין תאים.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = QtInteractor(self)
        self.plotter.setObjectName("squarePlotter")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        side = min(w, h)
        x = (w - side) // 2
        y = (h - side) // 2
        self.plotter.setGeometry(x, y, side, side)


class LiveViewsWindow(QWidget):
    """
    Manual positioning – Live views
    פריסה: 2 שורות × 3 טורים (6 תאים), כאשר 5 תצוגות מאוכלסות.
    כדי שכל האובייקטים יהיו "שווים" בגודל: כל תאי הגריד באותו stretch,
    וה-Plots עטופים ב-SquarePlotterWidget.
    """
    viewApplied = Signal(dict)

    def __init__(self, mesh_aligned: pv.PolyData, parent=None):
        super().__init__(parent)

        # ---- חלון ----
        self.setWindowTitle("manual positioning")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1200, 850)

        self._mesh = mesh_aligned
        self._ready = False
        self._syncing = False
        self._last_capture_dir: str | None = None

        self._dist_factor = 1.5

        # ---- מצב סיבוב ידני ----
        self._rotation_step_deg: float | None = None  # None = Free rotation (drag רגיל)
        self._click_rotate_enabled: bool = True

        # [תוספת] מצב תצוגת צירים
        self._axes_visible: bool = False

        # mappings
        self._plotter_to_key: Dict[QtInteractor, str] = {}
        self._cam_observer_tags: Dict[str, int] = {}
        self._cam_to_key: Dict[int, str] = {}

        # scale בסיסי גלובלי
        self._global_base_scale = self._compute_global_parallel_scale(self._mesh, margin=1.1)

        # ---- root layout ----
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setLayout(root)

        # =======================================================
        # splitter אופקי: משמאל כפתורים, מימין פלוטרים
        # =======================================================
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(4)
        root.addWidget(splitter, 1)

        # =======================================================
        # שמאל: פאנל כפתורים
        # =======================================================
        left_panel = QWidget()
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # כפתור נפתח לבחירת צעד סיבוב
        self._btn_rotate = ArtifactToolButton("free rotation", left_panel)
        self._btn_rotate.setPopupMode(QToolButton.InstantPopup)

        menu_rotate = QMenu(self._btn_rotate)
        self._act_free = QAction("Free rotation", menu_rotate)
        self._act_2 = QAction("2 degrees", menu_rotate)
        self._act_5 = QAction("5 degrees", menu_rotate)
        self._act_10 = QAction("10 degrees", menu_rotate)
        self._act_45 = QAction("45 degrees", menu_rotate)
        self._act_90 = QAction("90 degrees", menu_rotate)

        for act in (self._act_free, self._act_2, self._act_5, self._act_10, self._act_45, self._act_90):
            menu_rotate.addAction(act)

        self._act_free.triggered.connect(lambda: self._set_rotation_step(None))
        self._act_2.triggered.connect(lambda: self._set_rotation_step(2.0))
        self._act_5.triggered.connect(lambda: self._set_rotation_step(5.0))
        self._act_10.triggered.connect(lambda: self._set_rotation_step(10.0))
        self._act_45.triggered.connect(lambda: self._set_rotation_step(45.0))
        self._act_90.triggered.connect(lambda: self._set_rotation_step(90.0))

        self._btn_rotate.setMenu(menu_rotate)
        left_layout.addWidget(self._btn_rotate)

        # [תוספת] כפתור Show Axes
        self._btn_axes = ArtifactButton("show axes", left_panel)
        self._btn_axes.setCheckable(True)
        self._btn_axes.clicked.connect(self._toggle_axes_visibility)
        left_layout.addWidget(self._btn_axes)

        btn_reset = ArtifactButton("reset", left_panel)
        btn_reset.clicked.connect(self._on_reset_clicked)
        left_layout.addWidget(btn_reset)

        btn_apply = ArtifactButton("apply to main", left_panel)
        btn_apply.clicked.connect(self._on_apply_to_main_clicked)
        left_layout.addWidget(btn_apply)

        help_btn = ArtifactButton("help", left_panel)
        help_btn.clicked.connect(self.help_dialog)
        left_layout.addWidget(help_btn)

        left_layout.addStretch(1)

        # (אופציונלי) להגביל רוחב הפאנל השמאלי כדי לתת יותר מקום לימין
        left_panel.setMaximumWidth(240)

        # =======================================================
        # ימין: גריד 2×3 עם תאים שווים (כדי שכל האובייקטים יהיו שווים)
        # =======================================================
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid = QGridLayout(right_container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        for r in range(2):
            grid.setRowStretch(r, 1)
        for c in range(3):
            grid.setColumnStretch(c, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 1)  # שמאל
        splitter.setStretchFactor(1, 5)  # ימין

        # =======================================================
        # views spec + layout positions
        # =======================================================
        pretty_names = {
            "MC": "front",
            "MC_BACK": "back",
            "ML": "side",
            "TL": "top",
            "BR": "bottom",
        }

        # מבטים לפי views_spec
        from artifact_app.viewer.views_spec import get_views_spec
        views = get_views_spec()

        # פריסה 2×3:
        # שורה עליונה: TL (שמאל), [ריק], BR (ימין)
        # שורה תחתונה: MC, ML, MC_BACK
        self._layout_positions: dict[str, tuple[int, int]] = {
            "TL": (0, 0),
            "BR": (0, 2),
            "MC": (1, 0),
            "ML": (1, 1),
            "MC_BACK": (1, 2),
        }

        used_keys = [k for k in self._layout_positions.keys() if k in views]
        if not used_keys:
            raise RuntimeError("no common view keys between views_spec and layout positions")

        # storage
        self.plotters: Dict[str, QtInteractor] = {}
        self._base_angles: Dict[str, Dict[str, float]] = {}
        self._base_scales: Dict[str, float] = {}

        # יצירת תאים לפי סדר רשת (row-major) כדי שהקוד יהיה קריא
        for key, (r, c) in sorted(self._layout_positions.items(), key=lambda kv: (kv[1][0], kv[1][1])):
            v = views.get(key)
            if v is None:
                continue

            display_name = pretty_names.get(key, key)
            cell = self._make_cell(right_container, key, v, display_name)
            grid.addWidget(cell, r, c)

        # ---- רישום מאזינים לכל מצלמה (אחרי הבנייה) ----
        self._attach_camera_observers()
        self._ready = True

    # ======================================================================
    # UI helpers
    # ======================================================================
    def _set_rotation_step(self, step: float | None) -> None:
        """עדכון גודל צעד הסיבוב במעלות (או None לסיבוב חופשי)."""
        self._rotation_step_deg = step
        if step is None:
            self._btn_rotate.setText("rotation: free")
        else:
            self._btn_rotate.setText(f"rotation: {step:.0f}°")

    def _make_cell(self, parent: QWidget, key: str, v: dict, title_text: str) -> QWidget:
        """תא אחד: כותרת + SquarePlotterWidget (שמכיל QtInteractor)."""
        cell = QWidget(parent)
        cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        vbox = QVBoxLayout(cell)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)

        title = QLabel(title_text, cell)
        title.setStyleSheet("font-weight: 600;")
        vbox.addWidget(title, 0)

        wrap = SquarePlotterWidget(cell)
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        p = wrap.plotter
        p.setObjectName(f"plotter_{key}")
        p.set_background("#B8C2CC")
        p.add_mesh(self._mesh, color="silver")

        pos, foc, up = self._camera_from_angles(
            self._mesh,
            float(v.get("az", 0.0)),
            float(v.get("el", 0.0)),
            float(v.get("roll", 0.0)),
        )
        p.camera_position = (pos, foc, up)

        try:
            p.enable_parallel_projection()
        except Exception:
            pass

        self._set_parallel_scale(p, self._global_base_scale)

        # [תוספת] אם הצירים דלוקים, להציג אותם גם ביצירה מחדש (למשל בריסט)
        if getattr(self, "_axes_visible", False):
            self._update_plotter_axes(p, True)

        p.render()

        p.installEventFilter(self)
        self._plotter_to_key[p] = key

        self._base_angles[key] = {
            "az": float(v.get("az", 0.0)),
            "el": float(v.get("el", 0.0)),
            "roll": float(v.get("roll", 0.0)),
        }
        self._base_scales[key] = self._global_base_scale

        vbox.addWidget(wrap, 1)
        self.plotters[key] = p
        return cell

    # ======================================================================
    # [תוספת] Axes / Bounding Box Logic
    # ======================================================================
    def _toggle_axes_visibility(self, checked: bool):
        """
        מדליק או מכבה את קופסת התיחום (Bounding Box) בכל הפלוטרים.
        """
        self._axes_visible = checked
        text = "hide axes" if checked else "show axes"
        self._btn_axes.setText(text)

        for p in self.plotters.values():
            self._update_plotter_axes(p, checked)
            p.render()

    def _update_plotter_axes(self, p: QtInteractor, visible: bool):
        """
        מצייר או מסיר את תיבת התיחום.
        """
        if visible:
            p.show_bounds(
                grid='front',
                location='outer',
                all_edges=True,
                color="black",
                xtitle="", ytitle="", ztitle="",
                font_size=10
            )
        else:
            p.remove_bounds_axes()

    # ======================================================================
    # Camera observers
    # ======================================================================
    def _attach_camera_observers(self) -> None:
        """מצמיד ModifiedEvent לכל מצלמה וממפה camera->key לזיהוי המקור."""
        for key, tag in list(self._cam_observer_tags.items()):
            p = self.plotters.get(key)
            if p is None:
                continue
            cam = p.renderer.GetActiveCamera()
            try:
                cam.RemoveObserver(tag)
            except Exception:
                pass

        self._cam_observer_tags.clear()
        self._cam_to_key.clear()

        for key, p in self.plotters.items():
            cam = p.renderer.GetActiveCamera()
            self._cam_to_key[id(cam)] = key
            tag = cam.AddObserver("ModifiedEvent", self._on_any_camera_modified)
            self._cam_observer_tags[key] = tag

    # ======================================================================
    # Core sync: stable matrix-based sync
    # ======================================================================
    def _on_any_camera_modified(self, caller, event):
        if (not self._ready) or self._syncing:
            return

        key_src = self._cam_to_key.get(id(caller))
        if not key_src:
            return

        src_plotter = self.plotters.get(key_src)
        if not src_plotter:
            return

        F_base = self._frame_from_base(self._base_angles[key_src])  # 3x3
        fwd_cur, up_cur = self._read_fwd_up(src_plotter)
        F_cur = self._frame_from_fwd_up(fwd_cur, up_cur)  # 3x3

        R_delta = F_cur @ F_base.T

        scale_src_now = self._get_parallel_scale(src_plotter)
        scale_src_base = self._base_scales.get(key_src) or 1.0
        zoom_ratio = (scale_src_now / scale_src_base) if scale_src_base else 1.0

        self._syncing = True
        try:
            for key, p in self.plotters.items():
                if key == key_src:
                    continue

                F_tgt_base = self._frame_from_base(self._base_angles[key])
                F_tgt_new = R_delta @ F_tgt_base

                right, up_new, fwd_new = F_tgt_new[:, 0], F_tgt_new[:, 1], F_tgt_new[:, 2]
                pos, foc, upv = self._camera_from_fwd_up(self._mesh, fwd_new, up_new)
                p.camera_position = (pos, foc, upv)

                base_scale_k = self._base_scales.get(key) or 0.0
                if base_scale_k:
                    self._set_parallel_scale(p, zoom_ratio * base_scale_k)

                p.render()
        finally:
            self._syncing = False

    # ======================================================================
    # Camera math helpers
    # ======================================================================
    def _compute_global_parallel_scale(self, mesh: pv.PolyData, margin: float = 1.1) -> float:
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        span = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
        max_span = float(np.max(span)) or 1.0
        return 0.5 * max_span * float(margin)

    def _frame_from_base(self, base_angles: dict) -> np.ndarray:
        az = float(base_angles.get("az", 0.0))
        el = float(base_angles.get("el", 0.0))
        roll = float(base_angles.get("roll", 0.0))

        right, up, fwd = compute_view_dirs_from_azel(az, el)

        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        up = up / (np.linalg.norm(up) or 1.0)

        if roll:
            up = self._rotate_vec_around_axis(up, axis=fwd, degrees=roll)

        right = np.cross(up, fwd)
        right = right / (np.linalg.norm(right) or 1.0)
        up = np.cross(fwd, right)

        return np.column_stack([right, up, fwd])  # 3x3

    @staticmethod
    def _frame_from_fwd_up(fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = fwd / (np.linalg.norm(fwd) or 1.0)
        u = up / (np.linalg.norm(up) or 1.0)
        r = np.cross(u, f)
        r = r / (np.linalg.norm(r) or 1.0)
        u = np.cross(f, r)
        return np.column_stack([r, u, f])  # 3x3

    @staticmethod
    def _read_fwd_up(p: QtInteractor) -> tuple[np.ndarray, np.ndarray]:
        cam = p.renderer.GetActiveCamera()
        pos = np.array(cam.GetPosition(), float)
        foc = np.array(cam.GetFocalPoint(), float)
        up = np.array(cam.GetViewUp(), float)
        fwd = foc - pos
        return (
            fwd / (np.linalg.norm(fwd) or 1.0),
            up / (np.linalg.norm(up) or 1.0),
        )

    def _camera_from_fwd_up(self, mesh: pv.PolyData, fwd: np.ndarray, up: np.ndarray):
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        center = np.array([(xmax + xmin) / 2,
                           (ymax + ymin) / 2,
                           (zmax + zmin) / 2], float)
        span = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
        diag = float(np.linalg.norm(span)) or 1.0

        f = fwd / (np.linalg.norm(fwd) or 1.0)
        u = up / (np.linalg.norm(up) or 1.0)

        dist = self._dist_factor * diag
        pos = center - dist * f
        return tuple(pos.tolist()), tuple(center.tolist()), tuple(u.tolist())

    def _camera_from_angles(self, mesh: pv.PolyData, az: float, el: float, roll: float):
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
            up = self._rotate_vec_around_axis(up, axis=fwd, degrees=float(roll))
            right = np.cross(up, fwd)
            right = right / (np.linalg.norm(right) or 1.0)
            up = np.cross(fwd, right)

        dist = self._dist_factor * diag
        pos = center - dist * fwd
        return tuple(pos.tolist()), tuple(center.tolist()), tuple(up.tolist())

    @staticmethod
    def _rotate_vec_around_axis(v: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
        axis = axis / (np.linalg.norm(axis) or 1.0)
        theta = np.deg2rad(degrees)
        v_par = np.dot(v, axis) * axis
        v_perp = v - v_par
        w = np.cross(axis, v_perp)
        return v_par + v_perp * np.cos(theta) + w * np.sin(theta)

    # ======================================================================
    # ParallelScale helpers
    # ======================================================================
    @staticmethod
    def _get_parallel_scale(p: QtInteractor) -> float:
        try:
            return float(p.camera.GetParallelScale())
        except Exception:
            try:
                return float(p.renderer.GetActiveCamera().GetParallelScale())
            except Exception:
                return 0.0

    @staticmethod
    def _set_parallel_scale(p: QtInteractor, val: float) -> None:
        try:
            p.camera.SetParallelScale(float(val))
        except Exception:
            try:
                p.renderer.GetActiveCamera().SetParallelScale(float(val))
            except Exception:
                pass

    # ======================================================================
    # Dialogs / actions
    # ======================================================================
    def help_dialog(self) -> None:
        text = (
            "Manual Positioning – Live Views\n\n"
            "Use this window to manually adjust the orientation of the object "
            "from several synchronized views (front, back, side, top, bottom).\n\n"
            "Rotation mode\n"
            "Use the rotation drop-down on the left to choose between Free rotation "
            "and fixed rotation steps (2°, 5°, 10°, 45°, 90°).\n\n"
            "In Free rotation mode, drag with the left mouse button inside any view "
            "to rotate the object freely.\n\n"
            "In step rotation mode, click on the left or right side of a view to "
            "rotate the object left or right by the selected angle.\n\n"
            "Reset\n"
            "Click Reset to restore all views to their original default orientation and zoom.\n\n"
            "Apply to main\n"
            "Click Apply to main to send the current orientation of the front view "
            "to the main window, so that the main view uses this camera direction."
        )
        QMessageBox.information(self, "Manual Positioning – Help", text)

    def _on_collage_clicked(self) -> None:
        """
        שומר תמונה אחת שמרכיבה את כל הפלוטרים לפי הפריסה הנוכחית (2×3).
        """
        if not self.plotters:
            QMessageBox.warning(self, "אין תצוגות", "לא נמצאו פלוטרים לשמירה.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"liveviews_{ts}_collage.png"
        start_dir = getattr(self, "_last_capture_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "שמור קולאז׳ כ־PNG",
            os.path.join(start_dir, default_name),
            "PNG (*.png)",
        )
        if not path:
            return

        images: dict[str, np.ndarray] = {}
        max_w, max_h = 0, 0
        for key, p in self.plotters.items():
            try:
                img = p.screenshot(return_img=True)
                if img is None:
                    continue
                h, w = img.shape[:2]
                images[key] = img
                max_w = max(max_w, w)
                max_h = max(max_h, h)
            except Exception:
                pass

        if not images:
            QMessageBox.warning(self, "אין תמונות", "נכשלתי לקבל תמונות מהפלוטרים.")
            return

        rows, cols = 2, 3
        cell_w, cell_h = max_w, max_h
        canvas_w, canvas_h = cols * cell_w, rows * cell_h

        canvas = QImage(canvas_w, canvas_h, QImage.Format_ARGB32)
        canvas.fill(Qt.white)

        painter = QPainter(canvas)
        try:
            for key, (r, c) in self._layout_positions.items():
                arr = images.get(key)
                if arr is None:
                    continue
                qimg = self._numpy_to_qimage(arr)
                x = c * cell_w
                y = r * cell_h
                painter.drawImage(
                    x,
                    y,
                    qimg.scaled(cell_w, cell_h, Qt.KeepAspectRatio, Qt.SmoothTransformation),
                )
        finally:
            painter.end()

        ok = canvas.save(path, "PNG")
        if ok:
            self._last_capture_dir = os.path.dirname(path)
            QMessageBox.information(self, "נשמר", f"הקולאז׳ נשמר כ-\n{path}")
        else:
            QMessageBox.warning(self, "כשל בשמירה", "לא הצלחתי לשמור את הקובץ.")

    @staticmethod
    def _numpy_to_qimage(arr: np.ndarray) -> QImage:
        h, w = arr.shape[:2]
        if arr.shape[2] == 3:
            bgra = np.concatenate([arr, 255 * np.ones((h, w, 1), dtype=np.uint8)], axis=2)
        elif arr.shape[2] == 4:
            bgra = arr
        else:
            raise ValueError("תמיכה רק ב-RGB או RGBA")

        qimg = QImage(bgra.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return qimg.copy()

    # ======================================================================
    # Event filter: click-rotate in step mode
    # ======================================================================
    def eventFilter(self, obj, event):
        if not isinstance(obj, QtInteractor):
            return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if (self._rotation_step_deg is None) or (not self._click_rotate_enabled):
                return super().eventFilter(obj, event)

            w = obj.width()
            try:
                x = event.position().x()
            except AttributeError:
                x = event.x()

            step = float(self._rotation_step_deg)
            delta = (-step) if (x < w / 2) else (+step)

            self._rotate_plotter_yaw(obj, delta)
            return True

        return super().eventFilter(obj, event)

    def _rotate_plotter_yaw(self, plotter: QtInteractor, delta_deg: float) -> None:
        try:
            fwd, up = self._read_fwd_up(plotter)
        except Exception:
            return

        f = fwd / (np.linalg.norm(fwd) or 1.0)
        u = up / (np.linalg.norm(up) or 1.0)

        scale_now = self._get_parallel_scale(plotter)

        up_new = self._rotate_vec_around_axis(u, axis=f, degrees=delta_deg)

        pos, foc, upv = self._camera_from_fwd_up(self._mesh, f, up_new)
        plotter.camera_position = (pos, foc, upv)

        if scale_now:
            self._set_parallel_scale(plotter, scale_now)

        plotter.render()

    # ======================================================================
    # Reset / Apply
    # ======================================================================
    def _on_reset_clicked(self) -> None:
        if not self.plotters:
            return

        self._syncing = True
        try:
            for key, p in self.plotters.items():
                base = self._base_angles.get(key)
                if not base:
                    continue

                az = float(base.get("az", 0.0))
                el = float(base.get("el", 0.0))
                roll = float(base.get("roll", 0.0))

                pos, foc, up = self._camera_from_angles(self._mesh, az, el, roll)
                p.camera_position = (pos, foc, up)

                base_scale = self._base_scales.get(key)
                if base_scale:
                    self._set_parallel_scale(p, base_scale)

                # [תוספת] איפוס מצב צירים אם הם דלוקים
                if getattr(self, "_axes_visible", False):
                    self._update_plotter_axes(p, True)

                p.render()
        finally:
            self._syncing = False

    def _on_apply_to_main_clicked(self) -> None:
        print("[live_views] _on_apply_to_main_clicked")

        p = self.plotters.get("MC")
        if p is None:
            QMessageBox.warning(self, "No MC view", "לא נמצא פלוטר MC להעברה.")
            return

        fwd, up = self._read_fwd_up(p)
        fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])

        az = float(np.degrees(np.arctan2(fx, fy)))
        fz_clamped = max(-1.0, min(1.0, fz))
        el = float(np.degrees(np.arcsin(fz_clamped)))

        _, up0, f0 = compute_view_dirs_from_azel(az, el)
        up0 = up0 / (np.linalg.norm(up0) or 1.0)
        f0 = f0 / (np.linalg.norm(f0) or 1.0)

        u = up / (np.linalg.norm(up) or 1.0)
        cross_ = np.cross(up0, u)
        dot_ = float(np.dot(up0, u))
        dot_ = max(-1.0, min(1.0, dot_))
        sign = np.sign(np.dot(f0, cross_))
        theta = np.degrees(np.arccos(dot_)) * (sign if sign != 0 else 1.0)
        roll = float(theta)

        print(
            "[live_views] extracted angles:",
            f"az={az:.2f}, el={el:.2f}, roll={roll:.2f}",
            "fwd=", np.round(fwd, 3),
            "up=", np.round(up, 3),
        )

        params = {
            "view_key": "MC",
            "az": az,
            "el": el,
            "roll": roll,
            "fwd": (float(fwd[0]), float(fwd[1]), float(fwd[2])),
            "up": (float(up[0]), float(up[1]), float(up[2])),
        }

        print(f"[live_views] emit viewApplied: {params}")
        self.viewApplied.emit(params)