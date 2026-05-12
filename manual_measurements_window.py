# -*- coding: utf-8 -*-
# src/artifact_app/gui/manual_measurements_window.py
from __future__ import annotations
from PySide6.QtGui import QPixmap
import pandas as pd
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QFrame
)

from artifact_app.viewer.views_spec import get_views_spec
from artifact_app.viewer.main_view import set_view_azel

try:
    from pyvistaqt import QtInteractor
except Exception:
    QtInteractor = None


@dataclass
class MeasureResult:
    A: np.ndarray
    B: np.ndarray
    distance_2d: float
    distance_3d: float
    z_angle_2d_rad: float
    z_angle_3d_rad: float


class ManualMeasurementsWindow(QMainWindow):
    """
    חלון מדידות ידניות (בדומה ל-MATLAB):
    - סיבוב חופשי עם העכבר
    - בחירת שתי נקודות (A, B)
    - ציור קו המדידה + קו ייחוס מקווקו לציר ה-Z + קשת לזווית
    - הצגת המרחק והזווית ב-2D/3D
    - שמירת התוצאות לקובץ טקסט (.txt)
    - שמירת צילום מסך של התצוגה (.tif/.png/.jpg)

    הערה:
    חלון זה מניח שהמודל כבר ייושר במערכת הצירים לפי NORMAL positioning ("uzy").
    הוא אינו תומך ביישור מבוסס אינרציה.
    """

    # הוספנו positioning_mode לחתימה
    def __init__(
            self,
            mesh: pv.PolyData,
            *,
            path_name: str = "",
            object_name: str = "object",
            postype: str = "uzy",
            parent=None,
            positioning_mode: str = "Normal Positioning"
    ):
        super().__init__(parent)

        if QtInteractor is None:
            raise RuntimeError("pyvistaqt is not available. Install pyvistaqt to use QtInteractor.")

        self.setWindowTitle("Manual Measurements")
        self.resize(1100, 800)

        self.mesh = pv.PolyData(mesh).copy()
        self.path_name = path_name or ""
        self.object_name = object_name or "object"
        self.postype = postype or "uzy"
        self.positioning_mode = positioning_mode  # שמירת המצב

        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._res: Optional[MeasureResult] = None

        self._measure_mode: bool = False
        self._angle_visible: bool = True

        self._actor_mesh = None
        self._actor_A = None
        self._actor_B = None
        self._actor_dis = None
        self._actor_ref = None
        self._actor_arc = None

        self._figure_scale: int = 2

        central = QWidget(self)
        self.setCentralWidget(central)

        # שינוי הלייאאוט הראשי לאפס שוליים כדי שהכותרת תמתח מקצה לקצה
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- יצירת שורת הכותרת (Header) ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #d0d7e2; border-bottom: 1px solid #ccc;")
        header_frame.setFixedHeight(40)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 0, 15, 0)

        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")

        lbl_mode = QLabel(f"[{self.positioning_mode}]")
        lbl_mode.setStyleSheet("font-weight: bold; font-size: 12px; color: #555; border: none;")
        lbl_mode.setAlignment(Qt.AlignCenter)

        lbl_spacer = QLabel("")  # לאיזון הפריסה

        header_layout.addWidget(self.lbl_title, 1)
        header_layout.addWidget(lbl_mode, 1)
        header_layout.addWidget(lbl_spacer, 1)

        root.addWidget(header_frame, 0)

        # --- אזור התוכן המרכזי ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        root.addWidget(content_widget, 1)

        self.lbl_sub = QLabel("", central)
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setStyleSheet("font-size:12px; color:#000;")
        content_layout.addWidget(self.lbl_sub)

        self.plot = QtInteractor(central)
        self.plot.setFocusPolicy(Qt.StrongFocus)
        self.plot.setFocus()
        content_layout.addWidget(self.plot, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        content_layout.addLayout(btn_row)

        self.btn_restore = QPushButton("Restore front view ", central)
        self.btn_measure = QPushButton("Measure (pick A then B)", central)
        self.btn_toggle_angle = QPushButton("Hide angle (Z)", central)
        self.btn_save_data = QPushButton("Save to Excel", central)
        self.btn_save_fig = QPushButton("Save figure (.tif/.png)", central)

        btn_row.addWidget(self.btn_restore)
        btn_row.addWidget(self.btn_measure)
        btn_row.addWidget(self.btn_toggle_angle)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_save_data)
        btn_row.addWidget(self.btn_save_fig)

        self.btn_restore.clicked.connect(self._on_restore_view)
        self.btn_measure.clicked.connect(self._on_measure_clicked)
        self.btn_toggle_angle.clicked.connect(self._on_toggle_angle)
        self.btn_save_data.clicked.connect(self._on_save_data)
        self.btn_save_fig.clicked.connect(self._on_save_fig)

        self._init_scene()

    def _init_scene(self) -> None:
        pl = self.plot

        # 1. צבע הרקע המדויק מ-Live Views
        pl.set_background("#B8C2CC")
        pl.enable_parallel_projection()

        mesh_draw = self.mesh
        try:
            mesh_draw = mesh_draw.compute_normals(
                point_normals=True,
                cell_normals=False,
                splitting=False,
                auto_orient_normals=True,
                inplace=False,
            )
        except Exception:
            pass

        # 2. הוספת המודל בפשטות, בדיוק כמו ב-Live Views
        self._actor_mesh = pl.add_mesh(
            mesh_draw,
            color="silver",
            pickable=True,
        )

        # (מחקנו את כל הגדרות ה-pv.Light כדי שהתוכנה תשתמש בתאורה הטבעית והנקייה)

        self._update_titles()
        self._on_restore_view()
    def _update_titles(self) -> None:
        # עכשיו הכותרת מכילה רק את שם האובייקט (המצב מוצג באופן קבוע ב-lbl_mode)
        self.lbl_title.setText(f"Manual Measurements: {self.object_name}")
    def _on_restore_view(self) -> None:
        pl = self.plot

        views = get_views_spec()
        v = views.get("MC", {"az": 0.0, "el": 0.0, "roll": 0.0})
        az = float(v.get("az", 0.0))
        el = float(v.get("el", 0.0))
        roll = float(v.get("roll", 0.0))

        c = np.asarray(self.mesh.center, float)
        b = self.mesh.bounds
        diag = float(max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6))
        radius = 0.5 * diag

        dist_scale = 1.20
        r_used = max(1e-6, radius * dist_scale)

        pl.enable_parallel_projection()

        set_view_azel(
            pl,
            az=az,
            el=el,
            center=c,
            radius=r_used,
            ortho=True,
            margin=1.0,
            bounds=None,
        )

        if roll:
            try:
                pl.camera.Roll(float(roll))
            except Exception:
                pass

        try:
            pl.renderer.reset_camera_clipping_range()
        except Exception:
            pass

        pl.render()
        self._set_info("Rotate freely with the mouse. Click Measure to pick A and B.")
        self._disable_picking()

    def _on_measure_clicked(self) -> None:
        if self._measure_mode:
            self._measure_mode = False
            self.btn_measure.setText("Measure (pick A then B)")
            self.btn_restore.setEnabled(True)
            self.btn_save_fig.setEnabled(True)
            self._disable_picking()
            self._set_info("Measurement canceled.")

            # שחרור ההקפאה למקרה שביטלנו באמצע
            try:
                self.plot.iren.interactor.Enable()
            except AttributeError:
                pass
            return

        self._measure_mode = True
        self.btn_measure.setText("Cancel measure")
        self.btn_restore.setEnabled(True)
        self.btn_save_fig.setEnabled(False)

        # מוודאים שהתצוגה משוחררת כדי לבחור נקודות
        try:
            self.plot.iren.interactor.Enable()
        except AttributeError:
            pass

        self._clear_measurement()
        self._A = None
        self._B = None
        self._res = None

        self._set_info("Pick start point A on the object.")
        self._enable_picking()
    def _enable_picking(self) -> None:
        pl = self.plot

        try:
            pl.setFocus()
        except Exception:
            pass

        try:
            pl.disable_picking()
        except Exception:
            pass

        def _cb(*args, **kwargs):
            point = None

            for cand in args[:2]:
                try:
                    p = np.asarray(cand, float).reshape(3)
                    if np.isfinite(p).all():
                        point = p
                        break
                except Exception:
                    pass

            if point is None:
                try:
                    p = np.asarray(pl.picked_point, float).reshape(3)
                    if np.isfinite(p).all():
                        point = p
                except Exception:
                    point = None

            if point is None:
                return

            self._on_pick_point(point)

        try:
            pl.enable_surface_point_picking(
                callback=_cb,
                show_message=False,
                show_point=False,
                left_clicking=True,
            )
        except Exception:
            pl.enable_point_picking(
                callback=_cb,
                show_message=False,
                show_point=False,
                left_clicking=True,
            )

    def _disable_picking(self) -> None:
        pl = self.plot
        try:
            pl.disable_picking()
        except Exception:
            pass

    def _on_pick_point(self, point) -> None:
        if not self._measure_mode:
            return

        p = np.asarray(point, float).reshape(3)

        if self._A is None:
            self._A = p
            self._draw_marker("A", p)
            self._set_info("Pick end point B on the object.")
            return

        if self._B is None:
            if np.allclose(p, self._A):
                return
            self._B = p
            self._draw_marker("B", p)

            self._res = self._compute_result(self._A, self._B)
            self._draw_measurement_overlay(self._res)

            self._measure_mode = False
            self.btn_measure.setText("Measure (pick A then B)")

            # מנטרלים את כפתור השחזור ברגע שהמדידה מסתיימת
            self.btn_restore.setEnabled(False)
            self.btn_save_fig.setEnabled(True)

            self._disable_picking()
            self._update_result_text(self._res)
            self._set_info("Done. You can Save data / Save figure / Hide angle / Measure again.")

            # פונקציה מקומית קטנה להקפאה (פותר את בעיית ההזחות)
            def do_freeze():
                try:
                    self.plot.iren.interactor.Disable()
                except AttributeError:
                    pass

            # מפעילים את ההקפאה עם טיימר של 200 מילישניות - חסין מבאגים!
            QTimer.singleShot(200, do_freeze)
    def _compute_result(self, A: np.ndarray, B: np.ndarray) -> MeasureResult:
        A = np.asarray(A, float).reshape(3)
        B = np.asarray(B, float).reshape(3)

        if A[1] <= B[1]:
            a, b = A, B
        else:
            a, b = B, A

        dy2 = float(b[1] - a[1])
        dz2 = float(b[2] - a[2])
        dist2 = float(np.sqrt(dy2 * dy2 + dz2 * dz2))
        z_angle_2d = float(np.arcsin(dy2 / dist2)) if dist2 > 1e-12 else 0.0

        dx = float(B[0] - A[0])
        dy = float(B[1] - A[1])
        dz = float(B[2] - A[2])
        dist3 = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        num = float(np.sqrt(dx * dx + dy * dy))
        z_angle_3d = float(np.arcsin(num / dist3)) if dist3 > 1e-12 else 0.0

        return MeasureResult(
            A=A,
            B=B,
            distance_2d=dist2,
            distance_3d=dist3,
            z_angle_2d_rad=z_angle_2d,
            z_angle_3d_rad=z_angle_3d,
        )

    def _clear_measurement(self) -> None:
        pl = self.plot
        for a in (self._actor_A, self._actor_B, self._actor_dis, self._actor_ref, self._actor_arc):
            if a is not None:
                try:
                    pl.remove_actor(a)
                except Exception:
                    pass
        self._actor_A = self._actor_B = None
        self._actor_dis = self._actor_ref = self._actor_arc = None
        pl.render()

    def _project_to_overlay_plane(self, P: np.ndarray) -> tuple[np.ndarray, float]:
        pl = self.plot
        cam = pl.camera

        dop = np.asarray(cam.GetDirectionOfProjection(), float)
        dop /= (np.linalg.norm(dop) + 1e-12)

        cam_pos = np.asarray(cam.GetPosition(), float)

        bnd = self.mesh.bounds
        diag = float(max(bnd[1] - bnd[0], bnd[3] - bnd[2], bnd[5] - bnd[4], 1e-6))
        eps = 0.002 * diag

        pts = np.asarray(self.mesh.points, float)
        depths = (pts - cam_pos) @ dop
        depth_plane = float(np.min(depths)) + eps

        P = np.asarray(P, float).reshape(3)
        depth_P = float(np.dot(P - cam_pos, dop))
        Pp = P + (depth_plane - depth_P) * dop
        return Pp, diag

    def _draw_marker(self, which: str, p: np.ndarray) -> None:
        pl = self.plot

        Pp, diag = self._project_to_overlay_plane(p)
        r = 0.010 * diag

        sph = pv.Sphere(
            radius=r,
            center=tuple(Pp),
            theta_resolution=24,
            phi_resolution=24,
        )

        actor = pl.add_mesh(
            sph,
            color="red",
            smooth_shading=False,
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            pickable=False,
        )

        if which == "A":
            if self._actor_A is not None:
                try:
                    pl.remove_actor(self._actor_A)
                except Exception:
                    pass
            self._actor_A = actor
        else:
            if self._actor_B is not None:
                try:
                    pl.remove_actor(self._actor_B)
                except Exception:
                    pass
            self._actor_B = actor

        try:
            pl.renderer.reset_camera_clipping_range()
        except Exception:
            pass

        pl.render()

    def _draw_measurement_overlay(self, res: MeasureResult) -> None:
        pl = self.plot
        cam = pl.camera

        A = res.A
        B = res.B

        dop = np.asarray(cam.GetDirectionOfProjection(), float)
        dop_norm = float(np.linalg.norm(dop))
        if dop_norm < 1e-12:
            dop = np.array([1.0, 0.0, 0.0], float)
        else:
            dop /= dop_norm

        cam_pos = np.asarray(cam.GetPosition(), float)

        bnd = self.mesh.bounds
        diag = float(max(bnd[1] - bnd[0], bnd[3] - bnd[2], bnd[5] - bnd[4], 1e-6))
        eps = 0.002 * diag

        pts = np.asarray(self.mesh.points, float)
        depths = (pts - cam_pos) @ dop
        depth_plane = float(np.min(depths)) + eps

        def to_overlay_plane(P: np.ndarray) -> np.ndarray:
            P = np.asarray(P, float).reshape(3)
            depth_P = float(np.dot(P - cam_pos, dop))
            return P + (depth_plane - depth_P) * dop

        def _dashed_polyline(p0: np.ndarray, p1: np.ndarray, *, dash=0.06, gap=0.04, nmax=300) -> pv.PolyData:
            p0 = np.asarray(p0, float).reshape(3)
            p1 = np.asarray(p1, float).reshape(3)
            v = p1 - p0
            L = float(np.linalg.norm(v))
            if L < 1e-12:
                return pv.PolyData()

            u = v / L
            dash_len = dash * L
            gap_len = gap * L

            pts_out = []
            lines = []
            idx = 0
            tcur = 0.0
            k = 0
            while tcur < L and k < nmax:
                t0 = tcur
                t1 = min(L, tcur + dash_len)
                P0 = p0 + t0 * u
                P1 = p0 + t1 * u
                pts_out.append(P0)
                pts_out.append(P1)
                lines.extend([2, idx, idx + 1])
                idx += 2
                tcur = t1 + gap_len
                k += 1

            poly = pv.PolyData(np.array(pts_out, float))
            poly.lines = np.array(lines, np.int64)
            return poly

        z0 = float(bnd[4])
        zf = float(bnd[5])

        # --- התיקון החדש: מציאת הכיוון "ימינה" של המצלמה ומיון ---
        up = np.asarray(cam.GetViewUp(), float)
        up_norm = float(np.linalg.norm(up))
        up = up / up_norm if up_norm > 1e-12 else np.array([0.0, 0.0, 1.0], float)

        right = np.cross(dop, up)
        right_norm = float(np.linalg.norm(right))
        right = right / right_norm if right_norm > 1e-12 else np.array([1.0, 0.0, 0.0], float)

        if float(np.dot(A, right)) < float(np.dot(B, right)):
            a, b = A, B
        else:
            a, b = B, A
        # -----------------------------------------------------------

        a_p = to_overlay_plane(a)
        b_p = to_overlay_plane(b)

        dis_pts = np.vstack([a_p, b_p])
        dis_line = pv.lines_from_points(dis_pts, close=False)
        self._actor_dis = pl.add_mesh(
            dis_line,
            color="black",
            line_width=3,
            pickable=False,
            render_lines_as_tubes=True,
        )

        v = b_p - a_p
        dist2_vis = float(np.linalg.norm(v))
        if dist2_vis < 1e-12:
            pl.render()
            return
        dis_dir = v / dist2_vis

        z_ref = z0 if (b[2] < a[2]) else zf
        ref_end = to_overlay_plane(np.array([a[0], a[1], z_ref], float))

        ref_vec = ref_end - a_p
        ref_n = float(np.linalg.norm(ref_vec))
        if ref_n < 1e-12:
            pl.render()
            return
        ref_dir = ref_vec / ref_n

        # ... (קוד שמייצר את קו המרחק dist2_vis וכו') ...

        c = float(np.clip(abs(np.dot(ref_dir, dis_dir)), -1.0, 1.0))
        theta_vis = float(np.arccos(c))

        # --- הסרנו את ה- if self._angle_visible: ---
        ref0 = to_overlay_plane(np.array([a[0], a[1], z0], float))
        ref1 = to_overlay_plane(np.array([a[0], a[1], zf], float))

        ref_dash = _dashed_polyline(ref0, ref1, dash=0.04, gap=0.03)
        self._actor_ref = pl.add_mesh(
            ref_dash,
            color="black",
            line_width=1,
            pickable=False,
            render_lines_as_tubes=False,
        )
        # תוספת: הגדרת הנראות לפי מצב הכפתור
        self._actor_ref.SetVisibility(self._angle_visible)

        if theta_vis > 1e-6:
            d_world = min(dist2_vis, abs(a[2] - z_ref))
            r = 0.5 * d_world

            perp = dis_dir - float(np.dot(dis_dir, ref_dir)) * ref_dir
            perp_n = float(np.linalg.norm(perp))
            if perp_n > 1e-12:
                perp /= perp_n

                t = np.linspace(0.0, theta_vis, 31)
                arc_pts = (
                        a_p[None, :]
                        + (r * np.cos(t))[:, None] * ref_dir[None, :]
                        + (r * np.sin(t))[:, None] * perp[None, :]
                )

                arc = pv.lines_from_points(arc_pts, close=False)
                self._actor_arc = pl.add_mesh(
                    arc,
                    color="black",
                    line_width=2,
                    pickable=False,
                    render_lines_as_tubes=True,
                )
                # תוספת: הגדרת הנראות לפי מצב הכפתור
                self._actor_arc.SetVisibility(self._angle_visible)

        try:
            pl.renderer.reset_camera_clipping_range()
        except Exception:
            pass

        pl.render()
    def _update_result_text(self, res: MeasureResult) -> None:
        if self._angle_visible:
            ang2 = float(np.degrees(res.z_angle_2d_rad))
            ang3 = float(np.degrees(res.z_angle_3d_rad))
            self.lbl_sub.setText(
                " | ".join(
                    [
                        # שינוי מ- :.3f ל- :.2f
                        f"2D: distance={res.distance_2d:.2f} mm,  angle w/ Z={ang2:.2f}°",
                        f"3D: distance={res.distance_3d:.2f} mm,  angle w/ Z={ang3:.2f}°",
                    ]
                )
            )
        else:
            self.lbl_sub.setText(
                " | ".join(
                    [
                        # שינוי מ- :.3f ל- :.2f
                        f"2D: distance={res.distance_2d:.2f} mm",
                        f"3D: distance={res.distance_3d:.2f} mm",
                    ]
                )
            )
    def _set_info(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _on_toggle_angle(self) -> None:
        self._angle_visible = not self._angle_visible
        self.btn_toggle_angle.setText("Hide angle (Z)" if self._angle_visible else "Show angle (Z)")

        # רק משנים את הנראות במקום למחוק ולצייר מחדש
        if self._actor_ref is not None:
            self._actor_ref.SetVisibility(self._angle_visible)
        if self._actor_arc is not None:
            self._actor_arc.SetVisibility(self._angle_visible)

        if self._res is not None:
            self._update_result_text(self._res)

        self.plot.render()
    def _on_save_data(self) -> None:
        if self._res is None:
            QMessageBox.information(self, "No data", "No measurement yet. Click Measure and pick A, B.")
            return

        # שינוי סיומת ברירת המחדל ל-xlsx
        default_name = f"{self.object_name} - manual measurements.xlsx"
        default_dir = self.path_name if self.path_name else os.path.expanduser("~")
        default_path = os.path.join(default_dir, default_name)

        # שינוי סינון הקבצים בחלון השמירה לאקסל
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save data",
            default_path,
            "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        res = self._res
        A, B = res.A, res.B
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ארגון הנתונים במבנה של מילון (כל מפתח יהיה עמודה באקסל)
        new_data = {
            "Date & Time": ts,
            "Object Name": self.object_name,
            "Positioning": self.postype,
            "Point A (X)": round(A[0], 6),
            "Point A (Y)": round(A[1], 6),
            "Point A (Z)": round(A[2], 6),
            "Point B (X)": round(B[0], 6),
            "Point B (Y)": round(B[1], 6),
            "Point B (Z)": round(B[2], 6),
            "Distance 3D (mm)": round(res.distance_3d, 6),
            "Angle w/ Z (deg)": round(float(np.degrees(res.z_angle_3d_rad)), 6)
        }

        # יצירת טבלה (DataFrame) מהשורה החדשה
        df_new = pd.DataFrame([new_data])

        # בדיקה אם קובץ האקסל כבר קיים כדי להוסיף לו שורה (Append)
        if os.path.exists(path):
            try:
                df_existing = pd.read_excel(path)
                # שרשור הנתונים הישנים עם השורה החדשה
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                # אם יש שגיאה בקריאת הקובץ הקיים (למשל קובץ פגום), ניצור מחדש
                df_final = df_new
        else:
            df_final = df_new

        try:
            # שמירת הנתונים לקובץ האקסל (ללא עמודת אינדקסים)
            df_final.to_excel(path, index=False)
            QMessageBox.information(self, "Saved", f"Saved to Excel:\n{path}")
        except PermissionError:
            # טיפול בשגיאה נפוצה בה הקובץ פתוח כרגע בתוכנת Excel
            QMessageBox.critical(self, "Save Error", "Permission denied.\nPlease close the Excel file if it is currently open and try again.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save to Excel:\n{str(e)}")
    def _on_save_fig(self) -> None:
        default_dir = self.path_name if self.path_name else os.path.expanduser("~")
        default_name = f"{self.object_name} - manual measurements.png"
        default_path = os.path.join(default_dir, default_name)

        path, filt = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            default_path,
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return

        low = path.lower()
        if not (low.endswith(".png") or low.endswith(".tif") or low.endswith(".tiff") or low.endswith(
                ".jpg") or low.endswith(".jpeg")):
            if "TIFF" in filt:
                path += ".tif"
            elif "JPEG" in filt:
                path += ".jpg"
            else:
                path += ".png"

        try:
            self.plot.render()
        except Exception:
            pass

        pix = self.centralWidget().grab()

        ok = pix.save(path)
        if not ok:
            fallback = os.path.splitext(path)[0] + ".png"
            pix.save(fallback)
            QMessageBox.warning(
                self,
                "Saved with fallback",
                f"Could not save as the selected format.\nSaved as:\n{fallback}",
            )
            return

        QMessageBox.information(self, "Saved", f"Saved figure:\n{path}")