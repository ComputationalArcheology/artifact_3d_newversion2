# -*- coding: utf-8 -*-
# src/artifact_app/gui/process_object_window.py
# VERSION 11 - פרופורציות קבועות (Equate Subplots)
from __future__ import annotations
from artifact_app.gui.widgets import ArtifactButton
from PySide6.QtCore import QRect

import numpy as np
from typing import Optional, Dict, Any, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QAction, QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QPlainTextEdit,
    QSplitter, QToolButton, QMenu, QFileDialog, QMessageBox, QInputDialog
)

from artifact_app.processing.process_object_measurements import (
    compute_process_object_measurements,
    ProfileResult,
    CaliperSet,
    ProcessObjectMeasurements
)
from artifact_app.processing.center_of_mass import compute_area_weighted_centroid
from artifact_app.processing.export_utils import export_measurements_to_excel

from artifact_app.gui.mesh_viewer import InteractiveMeshViewer


class ProcessObjectWindow(QWidget):
    def __init__(self, mesh_aligned, mc_meta=None, ml_meta=None, parent=None, filename: str = "", positioning_mode: str = "Normal Positioning"):
        super().__init__(parent)

        self.mesh = mesh_aligned
        self.mc_meta = mc_meta or {}
        self.ml_meta = ml_meta or {}
        self.filename = filename or ""
        self.positioning_mode = positioning_mode
        self._parent = parent

        self.green_line_percent = 20
        self._last_measurements: Optional[ProcessObjectMeasurements] = None
        self._viewer_window = None
        self._cross_section_window = None
        self._export_views_window = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        header = QHBoxLayout()

        lbl_title = QLabel("Process Object")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setObjectName("LabelFilename")
        lbl_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lbl_mode = QLabel(f"[{self.positioning_mode}]")
        lbl_mode.setAlignment(Qt.AlignCenter)
        lbl_mode.setStyleSheet("font-weight:bold; font-size:12px; color:#555; background:#d0d7e2;")
        lbl_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lbl_filename = QLabel(self.filename)
        lbl_filename.setAlignment(Qt.AlignCenter)
        lbl_filename.setStyleSheet("font-weight:bold; font-size:14px; background:#d0d7e2;")
        lbl_filename.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_menu = ArtifactButton("Menu ▼")
        self.btn_menu.setMinimumWidth(120)

        menu = QMenu(self.btn_menu)

        self.act_open_3d = QAction("🔎 3D Probe (Coordinates)", self)
        self.act_open_3d.triggered.connect(self.open_3d_viewer)
        menu.addAction(self.act_open_3d)

        self.act_cross_section = QAction("✂️ Cross Section (Hatach)", self)
        self.act_cross_section.triggered.connect(self.open_cross_section)
        menu.addAction(self.act_cross_section)

        menu.addSeparator()

        self.act_set_green = QAction("📏 Set Green Line Height...", self)
        self.act_set_green.triggered.connect(self.set_green_line_height)
        menu.addAction(self.act_set_green)

        menu.addSeparator()

        self.act_export_views = QAction("🖼️ Export Views (5 Views + Scale)", self)
        self.act_export_views.triggered.connect(self.open_export_views)
        menu.addAction(self.act_export_views)

        self.act_export_excel = QAction("📊 Export to Excel", self)
        self.act_export_excel.triggered.connect(self.export_to_excel)
        menu.addAction(self.act_export_excel)

        self.act_capture = QAction("📷 Capture Screenshot", self)
        self.act_capture.triggered.connect(self.take_screenshot)
        menu.addAction(self.act_capture)

        self.btn_menu.setMenu(menu)

        header.addWidget(lbl_title, 2)
        header.addWidget(lbl_mode, 1)
        header.addWidget(lbl_filename, 2)
        header.addWidget(self.btn_menu, 0)

        root.addLayout(header)

        self.lbl_profile_xz = QLabel("XZ (Thickness View)")
        self.lbl_profile_xz.setAlignment(Qt.AlignCenter)
        self.lbl_profile_xz.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_profile_xz.setStyleSheet("background:white; border:1px solid #ccc;")

        self.txt_info = QPlainTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.txt_info.setStyleSheet(
            "padding:8px; background:white; border:1px solid #ccc; font-family: Consolas, monospace;")
        self.txt_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.lbl_profile_yz = QLabel("YZ (Width View)")
        self.lbl_profile_yz.setAlignment(Qt.AlignCenter)
        self.lbl_profile_yz.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_profile_yz.setStyleSheet("background:white; border:1px solid #ccc;")

        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.lbl_profile_xz)
        self.split.addWidget(self.txt_info)
        self.split.addWidget(self.lbl_profile_yz)
        self.split.setStretchFactor(0, 4)
        self.split.setStretchFactor(1, 2)
        self.split.setStretchFactor(2, 4)

        root.addWidget(self.split, 1)

        self.resize(1100, 720)

        def _after_layout():
            w = max(1, self.width())
            self.split.setSizes([int(0.4 * w), int(0.2 * w), int(0.4 * w)])
            self.update_measurements()

        QTimer.singleShot(0, _after_layout)

    def _build_profile_pixmap_from_result(
            self,
            title: str,
            prof: ProfileResult,
            *,
            width: int,
            height: int,
            mirror_x: bool = False,
            fixed_scale: float = None,
            com_2d: tuple[float, float] = None
    ) -> QPixmap:
        pix = QPixmap(max(1, width), max(1, height))
        pix.fill(Qt.white)

        outline = np.asarray(prof.outline_uv, float) if prof.outline_uv is not None else np.zeros((0, 2), float)
        if outline.shape[0] < 3:
            p = QPainter(pix)
            p.drawText(10, 20, "No contour data")
            p.end()
            return pix

        if mirror_x:
            outline = outline.copy()
            outline[:, 0] = -outline[:, 0]

        u = outline[:, 0]
        v = outline[:, 1]
        umin, umax = float(np.min(u)), float(np.max(u))
        vmin, vmax = float(np.min(v)), float(np.max(v))

        span_u = umax - umin
        span_v = vmax - vmin

        pad_u = span_u * 0.12
        pad_v = span_v * 0.12

        L_world = umin - pad_u
        R_world = umax + pad_u
        B_world = vmin - pad_v
        T_world = vmax + pad_v

        world_w = R_world - L_world
        world_h = T_world - B_world

        margin = 50
        draw_w = width - 2 * margin
        draw_h = height - 2 * margin

        if world_w < 1e-9: world_w = 1.0
        if world_h < 1e-9: world_h = 1.0

        # שימוש בקנה מידה קבוע (אחיד לשני החלונות) אם סופק, אחרת חישוב מקומי
        if fixed_scale is not None:
            scale = fixed_scale
        else:
            scale = min(draw_w / world_w, draw_h / world_h)

        pixel_cx = width / 2
        pixel_cy = height / 2
        world_cx = (L_world + R_world) / 2
        world_cy = (B_world + T_world) / 2

        def to_screen(uu, vv):
            x = pixel_cx + (uu - world_cx) * scale
            y = pixel_cy - (vv - world_cy) * scale
            return float(x), float(y)

        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        font = QFont("Arial", 8)
        p.setFont(font)
        fm = p.fontMetrics()

        p.setPen(QPen(Qt.lightGray, 1))

        box_left, box_top = to_screen(umin, vmax)
        box_right, box_bottom = to_screen(umax, vmin)

        frame_left, frame_top = to_screen(L_world, T_world)
        frame_right, frame_bottom = to_screen(R_world, B_world)

        axis_font = QFont("Arial", 7)
        p.setFont(axis_font)
        p.setPen(Qt.darkGray)

        def nice_ticks(vmin, vmax, num_ticks=5):
            range_val = vmax - vmin
            if range_val <= 0: return [vmin]
            rough_step = range_val / num_ticks
            magnitude = 10 ** np.floor(np.log10(rough_step))
            residual = rough_step / magnitude
            if residual > 5:
                nice_step = 10 * magnitude
            elif residual > 2:
                nice_step = 5 * magnitude
            elif residual > 1:
                nice_step = 2 * magnitude
            else:
                nice_step = magnitude
            start = np.ceil(vmin / nice_step) * nice_step
            ticks = []
            tick = start
            while tick <= vmax:
                ticks.append(tick)
                tick += nice_step
            return ticks

        x_ticks = nice_ticks(umin, umax, 7)
        y_pos_line = to_screen(0, vmin)[1]

        for tick in x_ticks:
            sx, sy = to_screen(tick, vmin)
            p.drawLine(int(sx), int(y_pos_line), int(sx), int(y_pos_line + 4))
            label = f"{tick:.0f}" if abs(tick) >= 1 or tick == 0 else f"{tick:.1f}"
            tw = fm.horizontalAdvance(label)
            p.drawText(int(sx - tw / 2), int(y_pos_line + 14), label)

        y_ticks = nice_ticks(vmin, vmax, 8)
        x_pos_line = to_screen(umin, 0)[0]

        for tick in y_ticks:
            sx, sy = to_screen(umin, tick)
            p.drawLine(int(x_pos_line - 4), int(sy), int(x_pos_line), int(sy))
            label = f"{tick:.0f}" if abs(tick) >= 1 or tick == 0 else f"{tick:.1f}"
            tw = fm.horizontalAdvance(label)
            p.drawText(int(x_pos_line - tw - 6), int(sy + 4), label)

        p.setFont(font)

        p.setPen(QPen(Qt.black, 2))
        poly_points = [(to_screen(outline[i, 0], outline[i, 1])) for i in range(len(outline))]
        for i in range(len(poly_points) - 1):
            p.drawLine(int(poly_points[i][0]), int(poly_points[i][1]),
                       int(poly_points[i + 1][0]), int(poly_points[i + 1][1]))
        if len(poly_points) > 2:
            p.drawLine(int(poly_points[-1][0]), int(poly_points[-1][1]),
                       int(poly_points[0][0]), int(poly_points[0][1]))

        D = prof.matlab_data
        occupied_rects = []

        def draw_text_smart(x, y_line, text, color=Qt.black, preferred_offset=-5):
            """
            מצייר טקסט ומוודא שלא יעלה על טקסטים שכבר צוירו.
            preferred_offset: שלילי = מעל הקו, חיובי = מתחת לקו.
            אם יש התנגשות, הטקסט ידלג לצד השני של הקו.
            """
            p.setPen(color)

            rect = fm.boundingRect(text)
            text_w = rect.width()
            text_h = rect.height()

            y_pos = y_line + preferred_offset

            left = int(x - text_w / 2)
            top = int(y_pos - text_h)

            current_rect = QRect(left, top, text_w, text_h)

            collision = False
            for occupied in occupied_rects:
                if current_rect.intersects(occupied.adjusted(-2, -2, 2, 2)):
                    collision = True
                    break

            if collision:
                if preferred_offset < 0:
                    new_offset = abs(preferred_offset) + text_h
                else:
                    new_offset = -preferred_offset - text_h / 2

                y_pos = y_line + new_offset
                top = int(y_pos - text_h)
                current_rect = QRect(left, top, text_w, text_h)

            p.drawText(int(x - text_w / 2), int(y_pos), text)
            occupied_rects.append(current_rect)

        def mirror_val(val):
            return -val if mirror_x else val

        if D:
            # קווי רוחב כחולים (חיצוניים)
            if D.psCalw is not None:
                p.setPen(QPen(Qt.blue, 1))
                x1_orig, z1, x2_orig, z2 = D.psCalw
                x1 = mirror_val(x1_orig)
                x2 = mirror_val(x2_orig)
                if mirror_x: x1, x2 = x2, x1
                meas_y = vmin - (vmax - vmin) * 0.05
                sx1, sy = to_screen(x1, meas_y)
                sx2, _ = to_screen(x2, meas_y)
                p.drawLine(int(sx1), int(sy), int(sx2), int(sy))
                p.drawLine(int(sx1), int(sy - 5), int(sx1), int(sy + 5))
                p.drawLine(int(sx2), int(sy - 5), int(sx2), int(sy + 5))
                draw_text_smart((sx1 + sx2) / 2, sy + 12, f"{D.calw:.1f}", Qt.blue, preferred_offset=5)

            # קווי גובה כחולים
            if D.psCalh is not None:
                p.setPen(QPen(Qt.blue, 1))
                _, y1, _, y2 = D.psCalh
                meas_x = umin - (umax - umin) * 0.05
                if mirror_x: meas_x = umax + (umax - umin) * 0.05
                sx, sy1 = to_screen(meas_x, y1)
                _, sy2 = to_screen(meas_x, y2)
                p.drawLine(int(sx), int(sy1), int(sx), int(sy2))
                p.drawLine(int(sx - 5), int(sy1), int(sx + 5), int(sy1))
                p.drawLine(int(sx - 5), int(sy2), int(sx + 5), int(sy2))
                p.save()
                p.translate(int(sx - 10), int((sy1 + sy2) / 2))
                p.rotate(-90)
                p.setPen(Qt.blue)
                p.drawText(0, 0, f"{D.calh:.1f}")
                p.restore()

            # מיתרים ירוקים (מצוירים לפני האדומים כדי שיהוו עוגן למניעת התנגשויות טקסט)
            pen_green = QPen(Qt.darkGreen, 2)

            def draw_green(p_start, val, off=15):
                if p_start is None or val <= 0: return
                p.setPen(pen_green)
                gx_orig, gy = p_start
                gx = mirror_val(gx_orig)
                gx_end = gx + (val if not mirror_x else -val)
                if mirror_x: gx, gx_end = gx_end, gx
                sgx1, sgy = to_screen(gx, gy)
                sgx2, _ = to_screen(gx_end, gy)
                p.drawLine(int(sgx1), int(sgy), int(sgx2), int(sgy))
                p.drawEllipse(int(sgx1) - 2, int(sgy) - 2, 4, 4)
                p.drawEllipse(int(sgx2) - 2, int(sgy) - 2, 4, 4)

                draw_text_smart((sgx1 + sgx2) / 2, sgy, f"{val:.1f}", Qt.darkGreen, preferred_offset=off)

            draw_green(D.pCord80p, D.cord80p, -8)
            draw_green(D.pCordwhh, D.cordwhh, 15)
            draw_green(D.pCord20p, D.cord20p, 15)

            # מיתרים אדומים (אופקי ואנכי מקסימליים)
            if D.pCordw is not None and D.cordw > 0:
                p.setPen(QPen(Qt.red, 2))
                rx_start_orig, ry = D.pCordw
                rx_start = mirror_val(rx_start_orig)
                rx_end = rx_start + (D.cordw if not mirror_x else -D.cordw)
                if mirror_x: rx_start, rx_end = rx_end, rx_start
                srx1, sry = to_screen(rx_start, ry)
                srx2, _ = to_screen(rx_end, ry)
                p.drawLine(int(srx1), int(sry), int(srx2), int(sry))
                p.drawEllipse(int(srx1) - 2, int(sry) - 2, 4, 4)
                p.drawEllipse(int(srx2) - 2, int(sry) - 2, 4, 4)

                draw_text_smart((srx1 + srx2) / 2, sry, f"{D.cordw:.1f}", Qt.red, preferred_offset=-5)

            if D.pCordh is not None and D.cordh > 0:
                p.setPen(QPen(Qt.red, 2))
                rx_orig, ry_start = D.pCordh
                rx = mirror_val(rx_orig)
                ry_end = ry_start - D.cordh
                srx, sry1 = to_screen(rx, ry_start)
                _, sry2 = to_screen(rx, ry_end)
                p.drawLine(int(srx), int(sry1), int(srx), int(sry2))
                p.drawEllipse(int(srx) - 2, int(sry1) - 2, 4, 4)
                p.drawEllipse(int(srx) - 2, int(sry2) - 2, 4, 4)
                p.save()
                p.translate(int(srx + 12), int((sry1 + sry2) / 2))
                p.rotate(-90)
                p.setPen(Qt.red)
                p.drawText(0, 0, f"{D.cordh:.1f}")
                p.restore()

                # ציור מרכז המסה (CoM) על גבי הפרופיל
                if com_2d is not None:
                    cx_val = mirror_val(com_2d[0])
                    cy_val = com_2d[1]

                    cx_px, cy_px = to_screen(cx_val, cy_val)

                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(255, 0, 255))
                    p.drawEllipse(int(cx_px) - 4, int(cy_px) - 4, 8, 8)

                    p.setPen(QColor(255, 0, 255))
                    p.setFont(QFont("Arial", 8, QFont.Bold))
                    p.drawText(int(cx_px) + 6, int(cy_px) + 4, "CoM ")

                p.setPen(Qt.black)
                p.setFont(QFont("Arial", 10, QFont.Bold))
                p.drawText(10, 15, title)
                p.end()
                return pix

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, '_resize_timer') and self._resize_timer is not None:
            self._resize_timer.stop()

        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_done)
        self._resize_timer.start(200)

    def _on_resize_done(self):
        self._resize_timer = None
        self.update_measurements()

    def set_green_line_height(self):
        val, ok = QInputDialog.getInt(
            self,
            "Set Green Line Height",
            "Select percentage from bottom/top (0-50):",
            self.green_line_percent,
            0, 50, 1
        )
        if ok:
            self.green_line_percent = val
            self.update_measurements()

    def _calculate_shared_scale(self, prof_xz, prof_yz, w1, h1, w2, h2) -> float:
        """
        מחשב קנה מידה אחיד שיתאים לשני הפרופילים במקביל, כדי לשמור על יחס תצוגה
        זהה לזה שב-MATLAB (אותו יחס פיקסל/מ"מ).
        """
        margin = 50

        outline_xz = np.asarray(prof_xz.outline_uv, float)
        if len(outline_xz) > 0:
            world_w1 = np.ptp(outline_xz[:, 0]) * 1.24
            world_h1 = np.ptp(outline_xz[:, 1]) * 1.24
            if world_w1 < 1e-9: world_w1 = 1.0
            if world_h1 < 1e-9: world_h1 = 1.0
            scale_xz = min((w1 - 2 * margin) / world_w1, (h1 - 2 * margin) / world_h1)
        else:
            scale_xz = 9999.0

        outline_yz = np.asarray(prof_yz.outline_uv, float)
        if len(outline_yz) > 0:
            world_w2 = np.ptp(outline_yz[:, 0]) * 1.24
            world_h2 = np.ptp(outline_yz[:, 1]) * 1.24
            if world_w2 < 1e-9: world_w2 = 1.0
            if world_h2 < 1e-9: world_h2 = 1.0
            scale_yz = min((w2 - 2 * margin) / world_w2, (h2 - 2 * margin) / world_h2)
        else:
            scale_yz = 9999.0

        return min(scale_xz, scale_yz)

    def update_measurements(self) -> None:
        if self.mesh is None or self.mesh.n_points == 0:
            self.txt_info.setPlainText("No mesh.")
            return

        w1 = max(1, self.lbl_profile_xz.width())
        h1 = max(1, self.lbl_profile_xz.height())
        w2 = max(1, self.lbl_profile_yz.width())
        h2 = max(1, self.lbl_profile_yz.height())

        m = compute_process_object_measurements(
            self.mesh,
            mc_meta=self.mc_meta or {},
            view_size_xz=(w1, h1),
            view_size_yz=(w2, h2),
            height_percent=float(self.green_line_percent)
        )

        self._last_measurements = m

        try:
            actual_com, _ = compute_area_weighted_centroid(self.mesh)
        except Exception:
            actual_com = np.array([0.0, 0.0, 0.0])

        unified_scale = self._calculate_shared_scale(m.profile_xz, m.profile_yz, w1, h1, w2, h2)

        title_xz = f"XZ (Thickness) - Lines @ {self.green_line_percent}%"
        title_yz = f"YZ (Width) - Lines @ {self.green_line_percent}%"

        pm_xz = self._build_profile_pixmap_from_result(
            title_xz, m.profile_xz, width=w1, height=h1, mirror_x=False, fixed_scale=unified_scale,
            com_2d=(actual_com[0], actual_com[2])
        )
        self.lbl_profile_xz.setPixmap(pm_xz)

        pm_yz = self._build_profile_pixmap_from_result(
            title_yz, m.profile_yz, width=w2, height=h2, mirror_x=True, fixed_scale=unified_scale,
            com_2d=(actual_com[1], actual_com[2])
        )
        self.lbl_profile_yz.setPixmap(pm_yz)

        lines = []
        lines.append("=== MEASUREMENTS ===")
        lines.append(f"Green Line Height: {self.green_line_percent}% / {100 - self.green_line_percent}%")
        lines.append(f"Mesh: {m.n_points} points")
        lines.append("")
        if m.bind3box and m.bind3box.dims:
            d = m.bind3box.dims
            lines.append("Bounding Box (Caliper):")
            lines.append(f"  X (Width):     {d['dx']:.1f}")
            lines.append(f"  Y (Depth):     {d['dy']:.1f}")
            lines.append(f"  Z (Height):    {d['dz']:.1f}")
            lines.append(f"  Volume:        {m.mesh_volume:.1f}")

            if m.bind3box.center_world is not None:
                bbox_center = m.bind3box.center_world
                dist_to_com = np.linalg.norm(bbox_center - actual_com)
                lines.append(f"  BBox Center:   ({bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f})")
                lines.append(f"  Dist. to CoM:  {dist_to_com:.2f}")

        lines.append("")
        lines.append(f"XZ Max Width (Horiz):  {m.profile_xz.max_chord:.1f}")
        if m.profile_xz.matlab_data:
            lines.append(f"XZ Max Length (Vert):  {m.profile_xz.matlab_data.cordh:.1f}")

        lines.append(f"YZ Max Width (Horiz):  {m.profile_yz.max_chord:.1f}")
        if m.profile_yz.matlab_data:
            lines.append(f"YZ Max Length (Vert):  {m.profile_yz.matlab_data.cordh:.1f}")

        lines.append("")

        if m.profile_xz.matlab_data:
            lines.append("XZ (Thickness):")
            D = m.profile_xz.matlab_data
            lines.append(f"  @{self.green_line_percent}%:   {D.cord20p:.1f}")
            lines.append(f"  @50%:   {D.cordwhh:.1f}")
            lines.append(f"  @{100 - self.green_line_percent}%:   {D.cord80p:.1f}")

        if m.profile_yz.matlab_data:
            lines.append("\nYZ (Width):")
            D = m.profile_yz.matlab_data
            lines.append(f"  @{self.green_line_percent}%:   {D.cord20p:.1f}")
            lines.append(f"  @50%:   {D.cordwhh:.1f}")
            lines.append(f"  @{100 - self.green_line_percent}%:   {D.cord80p:.1f}")
        self.txt_info.setPlainText("\n".join(lines))

    def open_3d_viewer(self):
        if self.mesh is None:
            QMessageBox.warning(self, "Error", "No mesh loaded.")
            return

        if self._viewer_window is None or not self._viewer_window.isVisible():
            self._viewer_window = InteractiveMeshViewer(self.mesh, title=f"3D Probe: {self.filename}")

        self._viewer_window.show()
        self._viewer_window.raise_()
        self._viewer_window.activateWindow()

    def open_cross_section(self):
        if self.mesh is None:
            QMessageBox.warning(self, "Error", "No mesh loaded.")
            return

        try:
            from artifact_app.gui.cross_section_window import CrossSectionWindow

            if self._cross_section_window is None or not self._cross_section_window.isVisible():
                # העברת self._last_measurements לטובת ציור הקווים בחלון החיתוך
                self._cross_section_window = CrossSectionWindow(
                    mesh=self.mesh,
                    measurements=self._last_measurements,
                    title=f"Cross Section: {self.filename}",
                    filename=self.filename,
                    parent=None
                )

            self._cross_section_window.show()
            self._cross_section_window.raise_()
            self._cross_section_window.activateWindow()

        except Exception as e:
            print(f"Error opening cross section: {e}")

    def open_export_views(self):
        if self.mesh is None:
            QMessageBox.warning(self, "Error", "No mesh loaded.")
            return

        try:
            from artifact_app.gui.live_five_views_widget import LiveFiveViewsWindow

            if self._export_views_window is None or not self._export_views_window.isVisible():
                self._export_views_window = LiveFiveViewsWindow(
                    mesh=self.mesh,
                    title=f"Export Views: {self.filename}",
                    filename=self.filename,
                )

            self._export_views_window.show()
            self._export_views_window.raise_()
            self._export_views_window.activateWindow()

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Module Not Found",
                f"LiveFiveViewsWindow module not found.\n\n"
                f"Please ensure live_five_views_widget.py is in the gui folder.\n\n"
                f"Error: {e}"
            )

    def export_to_excel(self):
        if self._last_measurements is None:
            QMessageBox.warning(self, "Error", "No measurements to export.")
            return

        try:
            from artifact_app.processing.center_of_mass import compute_area_weighted_centroid
            actual_com, _ = compute_area_weighted_centroid(self.mesh)
        except Exception:
            actual_com = None

        base_name = self.filename.replace(".wrl", "").replace(".ply", "").replace(".obj", "") or "object"
        default_name = base_name + "_data.xlsx"

        path, _ = QFileDialog.getSaveFileName(self, "Save Excel Report", default_name, "Excel Files (*.xlsx)")
        if path:
            success = export_measurements_to_excel(self._last_measurements, self.filename, path, actual_com)
            if success:
                QMessageBox.information(self, "Success", f"Saved: {path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save file.")

    def take_screenshot(self):
        pixmap = self.grab()
        base_name = self.filename.replace(".wrl", "").replace(".ply", "").replace(".obj", "") or "object"
        default_name = base_name + "_screenshot.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", default_name, "Images (*.png *.jpg)")
        if path:
            pixmap.save(path)