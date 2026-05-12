# -*- coding: utf-8 -*-
# src/artifact_app/gui/export_views_window.py
"""
Export Views Window - חלון לייצוא 5 מבטים.
משתמש ב-MatlabFiveViewsCanvas ו-render_views_pixmaps_by_sizes
כדי להבטיח תוצאה ויזואלית הזהה לחלון הראשי.
"""
from __future__ import annotations

from typing import Dict, Optional, Any
import pyvista as pv

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QPushButton, QFileDialog, QMessageBox, QLineEdit
)

from artifact_app.viewer.view_matlab_style import (
    MatlabFiveViewsCanvas,
    render_views_pixmaps_by_sizes,
)
from artifact_app.viewer.scale_bar import make_scale_pixmap
from artifact_app.viewer.views_spec import get_views_spec


class ExportViewsWindow(QWidget):
    def __init__(
            self,
            mesh: pv.PolyData,
            title: str = "Export Views",
            parent=None,
            filename: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1000, 850)

        self._mesh = mesh
        self._filename = filename
        self._object_name = filename.replace(".wrl", "").replace(".ply", "").replace(".obj", "") or "Object"

        self._last_meta: Dict[str, dict] = {}
        self._last_pixmaps: Dict[str, QPixmap] = {}

        self._build_ui()

        # רינדור רק לאחר שה-layout מוכן
        QTimer.singleShot(100, self._render_views)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        title_frame = QFrame()
        title_frame.setStyleSheet("background: #A7B4C6; border-radius: 3px;")
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(10, 4, 10, 4)

        lbl_name = QLabel(self._object_name)
        lbl_name.setAlignment(Qt.AlignCenter)
        lbl_name.setStyleSheet("font-weight: bold; font-size: 14px; color: #444;")
        title_layout.addWidget(lbl_name)

        layout.addWidget(title_frame)

        self.views_box = QFrame()
        self.views_box.setStyleSheet("background: #DCE1E8; border: none;")
        self.views_box.setFrameShape(QFrame.NoFrame)
        self.views_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        views_layout = QVBoxLayout(self.views_box)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(0)

        self.five_views_canvas = MatlabFiveViewsCanvas(self.views_box)
        self.five_views_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        views_layout.addWidget(self.five_views_canvas, 1)

        if self._mesh is not None:
            bounds = self._mesh.bounds
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            self.five_views_canvas.set_bbox(dx, dy, dz)

        self.labels = self.five_views_canvas.labels

        layout.addWidget(self.views_box, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_save_jpg = QPushButton("💾 Save as JPG")
        btn_save_jpg.clicked.connect(lambda: self._save_image("jpg"))
        btn_layout.addWidget(btn_save_jpg)

        btn_save_png = QPushButton("💾 Save as PNG")
        btn_save_png.clicked.connect(lambda: self._save_image("png"))
        btn_layout.addWidget(btn_save_png)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _render_views(self):
        if self._mesh is None:
            return

        sizes = self.five_views_canvas.get_view_sizes()

        bg_other = "#DCE1E8"
        background_by_key = {
            "ML": bg_other,
            "MC_BACK": bg_other,
            "TL": bg_other,
            "BR": bg_other,
        }

        pixmaps = render_views_pixmaps_by_sizes(
            self._mesh,
            sizes_by_key=sizes,
            background="white",
            color="silver",
            dist_scale=1.9,
            zoom_fact=1.0,
            multi_samples=8,
            supersample=4.0,
            background_by_key=background_by_key,
            meta_out=self._last_meta,
            views_spec=get_views_spec(),
        )

        self._last_pixmaps = {}
        for key, pix in pixmaps.items():
            lbl = self.labels.get(key)
            self._last_pixmaps[key] = pix

            if lbl is None:
                continue
            lbl.setPixmap(pix)

        self._update_scale()

    def _update_scale(self):
        lbl = self.labels.get("SCALE")
        if lbl is None:
            return

        w, h = lbl.width(), lbl.height()
        if w < 10 or h < 10:
            return

        ref_key = "ML"
        ref_meta = self._last_meta.get(ref_key)
        if not isinstance(ref_meta, dict):
            return
        if float(ref_meta.get("parallel_scale", 0.0)) <= 0:
            return

        try:
            import numpy as np
            center = np.asarray(ref_meta.get("center", self._mesh.center), float)
            right = np.asarray(ref_meta["right"], float)
            right /= (np.linalg.norm(right) + 1e-12)
            u = (self._mesh.points.astype(float) - center[None, :]) @ right
            obj_w_mm = float(u.max() - u.min())

            pm = make_scale_pixmap(
                w, h,
                ref_meta,
                target_frac=0.80,
                dpr=self.devicePixelRatioF(),
                s0_mm=obj_w_mm,
                debug=True,
                bg="#DCE1E8"
            )
            lbl.setPixmap(pm)
        except Exception as e:
            print(f"Scale bar error: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # רינדור מחדש בעת שינוי גודל החלון (עם עיכוב - debounce)
        if hasattr(self, '_resize_timer'):
            self._resize_timer.stop()

        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._render_views)
        self._resize_timer.start(200)

    def _save_image(self, fmt: str):
        name = self._object_name
        default_name = f"{name} - full patch.{fmt}"

        filter_str = f"{fmt.upper()} (*.{fmt})"
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save as {fmt.upper()}",
            default_name,
            filter_str
        )

        if not path:
            return

        # לכידת תמונת אזור התצוגה בלבד (ללא המסגרת של החלון כולו)
        pixmap = self.views_box.grab()

        if pixmap.save(path):
            QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "Failed to save image.")