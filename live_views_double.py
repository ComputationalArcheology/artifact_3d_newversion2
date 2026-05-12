# -*- coding: utf-8 -*-
# src/artifact_app/gui/live_views_window.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
import os

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtCore import Qt, QEvent, Signal, QRect, QTimer
from PySide6.QtGui import QImage, QPainter, QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QSizePolicy,
    QFileDialog, QMessageBox,
    QToolButton, QMenu,
)

from artifact_app.gui.widgets import ArtifactButton, ArtifactToolButton
from artifact_app.viewer.views_spec import get_views_spec
from artifact_app.viewer.main_view import compute_view_dirs_from_azel


# ============================================================
#  A) Matlab-like layout helpers (in-file)
# ============================================================
@dataclass
class NRect:
    x: float
    y: float
    w: float
    h: float


def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def _subplot33_positions_matlab_like(
    *,
    L: float = 0.01,
    R: float = 0.01,
    B: float = 0.01,
    T: float = 0.01,
    gx: float = 0.005,
    gy: float = 0.005,
) -> dict[int, NRect]:
    cw = (1.0 - L - R - 2.0 * gx) / 3.0
    ch = (1.0 - B - T - 2.0 * gy) / 3.0

    def cell(row_from_bottom: int, col: int) -> NRect:
        x = L + col * (cw + gx)
        y = B + row_from_bottom * (ch + gy)
        return NRect(x, y, cw, ch)

    return {
        1: cell(2, 0),
        4: cell(1, 0),
        5: cell(1, 1),
        6: cell(1, 2),
        8: cell(0, 1),  # SCALE
        9: cell(0, 2),
    }


def compute_matlab_5rects_normalized(dx: float, dy: float, dz: float) -> dict[str, NRect]:
    # ratios
    rxy = _safe_div(dx, dy, 1.0)
    ryz = _safe_div(dy, dz, 1.0)
    rzx = _safe_div(dz, dx, 1.0)

    r = {
        1: rxy,
        4: _safe_div(1.0, ryz, 1.0),
        5: rzx,
        6: _safe_div(1.0, ryz, 1.0),
        9: rxy,
    }

    pos = _subplot33_positions_matlab_like()
    qos = {i: NRect(pos[i].x, pos[i].y, pos[i].w, pos[i].h) for i in (1, 4, 5, 6, 9)}

    # fit each rect to aspect (height/width) = r[i]
    for i, p in qos.items():
        rt = _safe_div(p.h, p.w, 1.0)
        if rt > r[i]:
            qos[i].h = p.w * r[i]
        else:
            qos[i].w = _safe_div(p.h, r[i], p.w)

    # equalize heights in row2 by 5
    refv5 = qos[5].h
    for i in (4, 6):
        q = qos[i]
        rat = _safe_div(refv5, q.h, 1.0)
        qos[i] = NRect(q.x, q.y, q.w * rat, q.h * rat)

    # equalize widths in col1/col3
    refh4 = qos[4].w
    q1 = qos[1]
    rat = _safe_div(refh4, q1.w, 1.0)
    qos[1] = NRect(q1.x, q1.y, q1.w * rat, q1.h * rat)

    refh6 = qos[6].w
    q9 = qos[9]
    rat = _safe_div(refh6, q9.w, 1.0)
    qos[9] = NRect(q9.x, q9.y, q9.w * rat, q9.h * rat)

    # stick compute scale
    e1 = pos[4].x
    e2 = pos[6].x + pos[6].w
    q1x = qos[4].x
    q2x = q1x + qos[4].w + qos[5].w + qos[6].w
    sch = _safe_div((q2x - q1x), (e2 - e1), 1.0)

    e3 = pos[9].y
    e4 = pos[1].y + pos[1].h
    q3y = qos[9].y
    q4y = qos[9].y + qos[9].h + qos[6].h + qos[1].h
    scv = _safe_div((q4y - q3y), (e4 - e3), 1.0)

    sc = max(sch, scv)

    ros = {i: NRect(0, 0, 0, 0) for i in (1, 4, 5, 6, 9)}
    ros[1].x = (1.0 - (q2x - q1x) / sc) / 2.0
    ros[1].w = qos[1].w / sc
    ros[1].h = qos[1].h / sc

    ros[4].x = ros[1].x
    ros[4].w = qos[4].w / sc
    ros[4].h = qos[4].h / sc

    ros[5].x = ros[4].x + ros[4].w
    ros[5].w = qos[5].w / sc
    ros[5].h = qos[5].h / sc

    ros[6].x = ros[5].x + ros[5].w
    ros[6].w = qos[6].w / sc
    ros[6].h = qos[6].h / sc

    ros[9].x = ros[6].x
    ros[9].w = qos[9].w / sc
    ros[9].h = qos[9].h / sc

    spanh = (q2x - q1x) / sc
    spanv = (q4y - q3y) / sc

    if spanh > spanv:
        mL = ros[4].x
        mR = 1.0 - mL - spanh
        seph = (mL + mR) / 4.0

        ros[4].x -= seph
        ros[1].x -= seph
        ros[6].x += seph
        ros[9].x += seph

        tbt = (1.0 - spanv - 2.0 * seph) / 2.0
        ros[9].y = tbt
        ros[4].y = ros[9].y + ros[9].h + seph
        ros[5].y = ros[4].y
        ros[6].y = ros[4].y
        ros[1].y = 1.0 - tbt - ros[1].h
    else:
        mB = ros[9].y
        mT = 1.0 - mB - spanv
        sepv = (mB + mT) / 4.0

        ros[9].y = sepv
        ros[4].y = ros[9].y + ros[9].h + sepv
        ros[5].y = ros[4].y
        ros[6].y = ros[4].y
        ros[1].y = ros[6].y + ros[6].h + sepv

        tbt = (1.0 - spanh - 2.0 * sepv) / 2.0
        ros[4].x = tbt
        ros[1].x = tbt
        ros[9].x += sepv
        ros[6].x += sepv

    return {
        "BR": ros[1],
        "MC": ros[4],
        "ML": ros[5],
        "MC_BACK": ros[6],
        "TL": ros[9],
    }


# ============================================================
#  B) Matlab-like Live Canvas (in-file)
# ============================================================
class MatlabLiveViewsCanvas(QWidget):
    """
    קנבס שממקם 5 תאי live-view בפריסה מטלבית באמצעות setGeometry.
    כל תא מכיל:
      - QLabel קטן (כותרת)
      - QtInteractor (פלוטר)
    """
    layoutChanged = Signal()

    VIEW_KEYS = ("TL", "ML", "MC", "MC_BACK", "BR")

    def __init__(self, mesh: pv.PolyData, pretty_names: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self._mesh = mesh
        self._pad = 12

        self.pretty_names = pretty_names or {
            "MC": "front",
            "MC_BACK": "back",
            "ML": "side",
            "TL": "top",
            "BR": "bottom",
        }

        # bbox שממנו נגזרת הפריסה
        xmin, xmax, ymin, ymax, zmin, zmax = self._mesh.bounds
        self._bbox_dx = float(xmax - xmin)
        self._bbox_dy = float(ymax - ymin)
        self._bbox_dz = float(zmax - zmin)

        self.cells: dict[str, QWidget] = {}
        self.titles: dict[str, QLabel] = {}
        self.plotters: dict[str, QtInteractor] = {}

        for key in self.VIEW_KEYS:
            cell = QWidget(self)
            cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            vbox = QVBoxLayout(cell)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(4)

            title = QLabel(self.pretty_names.get(key, key), cell)
            title.setStyleSheet("font-weight: 600;")
            title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            p = QtInteractor(cell)
            p.setObjectName(f"plotter_{key}")
            p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            vbox.addWidget(title, 0)
            vbox.addWidget(p, 1)

            self.cells[key] = cell
            self.titles[key] = title
            self.plotters[key] = p

        self._layout_matlab_like()

    def get_plotter_size(self, key: str) -> tuple[int, int]:
        p = self.plotters.get(key)
        if p is None:
            return (1, 1)
        return (max(1, p.width()), max(1, p.height()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_matlab_like()
        self.layoutChanged.emit()

    def _layout_matlab_like(self) -> None:
        W, H = self.width(), self.height()
        pad = int(self._pad)

        inner_w = max(1, W - 2 * pad)
        inner_h = max(1, H - 2 * pad)

        def to_qrect(nr: NRect) -> QRect:
            x = pad + int(round(nr.x * inner_w))
            w = int(round(nr.w * inner_w))
            # NRect.y מוגדר ממטה; Qt צריך y מלמעלה
            y_top = 1.0 - (nr.y + nr.h)
            y = pad + int(round(y_top * inner_h))
            h = int(round(nr.h * inner_h))
            return QRect(x, y, max(1, w), max(1, h))

        dx, dy, dz = self._bbox_dx, self._bbox_dy, self._bbox_dz
        if not (dx and dy and dz):
            pos = _subplot33_positions_matlab_like()
            rects = {
                "TL": to_qrect(pos[9]),
                "ML": to_qrect(pos[5]),
                "MC": to_qrect(pos[4]),
                "MC_BACK": to_qrect(pos[6]),
                "BR": to_qrect(pos[1]),
            }
        else:
            nrects = compute_matlab_5rects_normalized(dx, dy, dz)
            rects = {k: to_qrect(v) for k, v in nrects.items() if k in self.VIEW_KEYS}

        for key, rect in rects.items():
            self.cells[key].setGeometry(rect)
            self.cells[key].show()


# ============================================================
#  C) LiveViewsWindow (main)
# ============================================================
class LiveViewsWindow(QWidget):
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
        self._rotation_step_deg: float | None = None  # None = Free rotation
        self._click_rotate_enabled: bool = True
        self._plotter_to_key: Dict[QtInteractor, str] = {}

        # ---- cache ל-fit-scale (מהיר) ----
        self._mesh_center = self._compute_mesh_center(self._mesh)
        self._pts_sample = self._sample_points(self._mesh, max_n=60000)
        self._last_fit_by_key: Dict[str, float] = {}

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

        # -------- שמאל: פאנל כפתורים --------
        left_panel = QWidget(splitter)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # כפתור נפתח לבחירת צעד סיבוב
        self._btn_rotate = ArtifactToolButton("rotation: free", left_panel)
        self._btn_rotate.setPopupMode(QToolButton.InstantPopup)

        menu_rotate = QMenu(self._btn_rotate)
        self._act_free = QAction("Free rotation", menu_rotate)
        self._act_2 = QAction("2 degrees", menu_rotate)
        self._act_5 = QAction("5 degrees", menu_rotate)
        self._act_10 = QAction("10 degrees", menu_rotate)
        self._act_45 = QAction("45 degrees", menu_rotate)
        self._act_90 = QAction("90 degrees", menu_rotate)

        menu_rotate.addAction(self._act_free)
        menu_rotate.addAction(self._act_2)
        menu_rotate.addAction(self._act_5)
        menu_rotate.addAction(self._act_10)
        menu_rotate.addAction(self._act_45)
        menu_rotate.addAction(self._act_90)

        self._act_free.triggered.connect(lambda: self._set_rotation_step(None))
        self._act_2.triggered.connect(lambda: self._set_rotation_step(2.0))
        self._act_5.triggered.connect(lambda: self._set_rotation_step(5.0))
        self._act_10.triggered.connect(lambda: self._set_rotation_step(10.0))
        self._act_45.triggered.connect(lambda: self._set_rotation_step(45.0))
        self._act_90.triggered.connect(lambda: self._set_rotation_step(90.0))

        self._btn_rotate.setMenu(menu_rotate)
        left_layout.addWidget(self._btn_rotate)

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

        # -------- ימין: Canvas מטלבי --------
        top_container = QWidget(splitter)
        top_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setSpacing(0)

        pretty_names = {
            "MC": "front",
            "MC_BACK": "back",
            "ML": "side",
            "TL": "top",
            "BR": "bottom",
        }

        self.canvas = MatlabLiveViewsCanvas(self._mesh, pretty_names=pretty_names, parent=top_container)
        top_layout.addWidget(self.canvas, 1)

        splitter.addWidget(left_panel)  # index 0  ← כפתורים
        splitter.addWidget(top_container)  # index 1  ← תצוגות

        splitter.setStretchFactor(0, 1)  # 20%
        splitter.setStretchFactor(1, 4)  # 80%

        # אופציונלי אבל מאוד עוזר: קובע גדלים התחלתיים בפועל
        splitter.setSizes([250, 950])  # תתאים לערכים שלך

        # פלוטרים מתוך הקנבס
        self.plotters: Dict[str, QtInteractor] = self.canvas.plotters

        # ===== views / base info =====
        views = get_views_spec()

        self._base_angles: Dict[str, Dict[str, float]] = {}
        self._base_scales: Dict[str, float] = {}
        self._cam_observer_tags: Dict[str, int] = {}
        self._cam_to_key: Dict[int, str] = {}

        # ===== init plotters =====
        for key, p in self.plotters.items():
            v = views.get(key, {}) if isinstance(views, dict) else {}

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

            p.render()

            p.installEventFilter(self)
            self._plotter_to_key[p] = key

            self._base_angles[key] = {
                "az": float(v.get("az", 0.0)),
                "el": float(v.get("el", 0.0)),
                "roll": float(v.get("roll", 0.0)),
            }

            # base_scale = FIT של התא (ולא global)
            F_base = self._frame_from_base(self._base_angles[key])
            fit0 = self._fit_parallel_scale(
                right=F_base[:, 0],
                up=F_base[:, 1],
                window_size=self.canvas.get_plotter_size(key),
                margin=1.02,
            )
            self._set_parallel_scale(p, fit0)
            self._base_scales[key] = fit0
            self._last_fit_by_key[key] = fit0

            p.render()

        # ---- רישום מאזינים לכל מצלמה (אחרי הבנייה) ----
        self._attach_camera_observers()

        # שמירה על zoom בעת resize של התאים
        self.canvas.layoutChanged.connect(self._on_canvas_layout_changed)

        self._ready = True

        # תיקון ראשוני אחרי שה-Qt סידר גדלים אמיתיים
        QTimer.singleShot(0, self._on_canvas_layout_changed)

    # =======================================================
    # Canvas resize -> preserve zoom
    # =======================================================
    def _on_canvas_layout_changed(self) -> None:
        if not self._ready or not self.plotters or self._syncing:
            return

        self._syncing = True
        try:
            for key, p in self.plotters.items():
                fwd, up = self._read_fwd_up(p)
                F_cur = self._frame_from_fwd_up(fwd, up)

                scale_now = self._get_parallel_scale(p)

                fit_new = self._fit_parallel_scale(
                    right=F_cur[:, 0],
                    up=F_cur[:, 1],
                    window_size=self.canvas.get_plotter_size(key),
                    margin=1.02,
                )

                fit_old = self._last_fit_by_key.get(key, fit_new)
                zoom_factor = (scale_now / fit_old) if fit_old else 1.0

                self._set_parallel_scale(p, zoom_factor * fit_new)
                self._last_fit_by_key[key] = fit_new

                p.render()
        finally:
            self._syncing = False

    # =======================================================
    # Camera observers: all plotters lead
    # =======================================================
    def _attach_camera_observers(self):
        """מצמיד ModifiedEvent לכל מצלמה וממפה camera->key לזיהוי המקור."""
        for key, tag in list(self._cam_observer_tags.items()):
            cam = self.plotters[key].renderer.GetActiveCamera()
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

    def _set_rotation_step(self, step: float | None) -> None:
        self._rotation_step_deg = step
        if step is None:
            self._btn_rotate.setText("rotation: free")
        else:
            self._btn_rotate.setText(f"rotation: {step:.0f}°")

    # =======================================================
    # Stable sync using zoom_factor + fit_scale
    # =======================================================
    def _on_any_camera_modified(self, caller, event):
        if not self._ready or self._syncing:
            return

        key_src = self._cam_to_key.get(id(caller))
        if not key_src:
            return

        src_plotter = self.plotters.get(key_src)
        if not src_plotter:
            return

        # 1) מסגרות בסיס ונוכחית של המקור
        F_base = self._frame_from_base(self._base_angles[key_src])   # 3x3
        fwd_cur, up_cur = self._read_fwd_up(src_plotter)
        F_cur = self._frame_from_fwd_up(fwd_cur, up_cur)

        # 2) דלתא רוטציה
        R_delta = F_cur @ F_base.T

        # 3) zoom_factor יציב: scale_now / fit_now
        scale_src_now = self._get_parallel_scale(src_plotter)
        fit_src_now = self._fit_parallel_scale(
            right=F_cur[:, 0],
            up=F_cur[:, 1],
            window_size=self.canvas.get_plotter_size(key_src),
            margin=1.02,
        )
        zoom_factor = (scale_src_now / fit_src_now) if fit_src_now else 1.0
        self._last_fit_by_key[key_src] = fit_src_now

        # 4) החלה על כל השאר
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

                fit_tgt = self._fit_parallel_scale(
                    right=right,
                    up=up_new,
                    window_size=self.canvas.get_plotter_size(key),
                    margin=1.02,
                )
                self._set_parallel_scale(p, zoom_factor * fit_tgt)
                self._last_fit_by_key[key] = fit_tgt

                p.render()
        finally:
            self._syncing = False

    # =======================================================
    # Fit-scale helpers
    # =======================================================
    def _compute_mesh_center(self, mesh: pv.PolyData) -> np.ndarray:
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        return np.array([(xmax + xmin) * 0.5,
                         (ymax + ymin) * 0.5,
                         (zmax + zmin) * 0.5], dtype=float)

    def _sample_points(self, mesh: pv.PolyData, max_n: int = 60000) -> np.ndarray:
        pts = np.asarray(mesh.points, dtype=float)
        if pts.shape[0] <= max_n:
            return pts
        step = max(1, pts.shape[0] // max_n)
        return pts[::step].copy()

    def _fit_parallel_scale(
        self,
        *,
        right: np.ndarray,
        up: np.ndarray,
        window_size: tuple[int, int],
        margin: float = 1.02
    ) -> float:
        w, h = window_size
        w = max(1, int(w))
        h = max(1, int(h))
        aspect = w / h

        right = right / (np.linalg.norm(right) + 1e-12)
        up = up / (np.linalg.norm(up) + 1e-12)

        d = self._pts_sample - self._mesh_center[None, :]
        r = d @ right
        u = d @ up

        h_span = float(r.max() - r.min())
        v_span = float(u.max() - u.min())

        base_ps = 0.5 * max(v_span, h_span / max(aspect, 1e-6), 1e-6)
        return float(base_ps * float(margin))

    # =======================================================
    # Camera math helpers
    # =======================================================
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
        up = up / (np.linalg.norm(up) or 1.0)

        return np.column_stack([right, up, fwd])

    @staticmethod
    def _frame_from_fwd_up(fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = fwd / (np.linalg.norm(fwd) or 1.0)
        u = up / (np.linalg.norm(up) or 1.0)
        r = np.cross(u, f)
        r = r / (np.linalg.norm(r) or 1.0)
        u = np.cross(f, r)
        u = u / (np.linalg.norm(u) or 1.0)
        return np.column_stack([r, u, f])

    @staticmethod
    def _read_fwd_up(p: QtInteractor) -> tuple[np.ndarray, np.ndarray]:
        cam = p.renderer.GetActiveCamera()
        pos = np.array(cam.GetPosition(), float)
        foc = np.array(cam.GetFocalPoint(), float)
        up = np.array(cam.GetViewUp(), float)
        fwd = foc - pos
        return (fwd / (np.linalg.norm(fwd) or 1.0),
                up / (np.linalg.norm(up) or 1.0))

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
            up = up / (np.linalg.norm(up) or 1.0)

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

    # =======================================================
    # ParallelScale helpers
    # =======================================================
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

    # =======================================================
    # Help dialog
    # =======================================================
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

    # =======================================================
    # Event filter: step rotation clicks
    # =======================================================
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

    # =======================================================
    # Reset
    # =======================================================
    def _on_reset_clicked(self):
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

                # fit-scale בסיסי לפי גודל התא הנוכחי
                F_base = self._frame_from_base(base)
                fit0 = self._fit_parallel_scale(
                    right=F_base[:, 0],
                    up=F_base[:, 1],
                    window_size=self.canvas.get_plotter_size(key),
                    margin=1.02,
                )
                self._set_parallel_scale(p, fit0)
                self._base_scales[key] = fit0
                self._last_fit_by_key[key] = fit0

                p.render()
        finally:
            self._syncing = False

    # =======================================================
    # Apply-to-main
    # =======================================================
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

    # =======================================================
    # Optional: collage saver (kept if you want later)
    # =======================================================
    def _on_collage_clicked(self):
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
            "PNG (*.png)"
        )
        if not path:
            return

        images = {}
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

        # layout 3x3 סטנדרטי (רק לשמירה)
        positions = {
            "TL": (0, 0),
            "MC": (1, 0),
            "ML": (1, 1),
            "MC_BACK": (1, 2),
            "BR": (2, 2),
        }
        rows, cols = 3, 3
        cell_w, cell_h = max_w, max_h
        canvas_w, canvas_h = cols * cell_w, rows * cell_h

        canvas = QImage(canvas_w, canvas_h, QImage.Format_ARGB32)
        canvas.fill(Qt.white)

        painter = QPainter(canvas)
        try:
            for key, (r, c) in positions.items():
                arr = images.get(key)
                if arr is None:
                    continue
                qimg = self._numpy_to_qimage(arr)
                x = c * cell_w
                y = r * cell_h
                painter.drawImage(x, y, qimg.scaled(cell_w, cell_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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