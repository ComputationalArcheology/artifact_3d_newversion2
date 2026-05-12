# src/artifact_app/gui/crop_window.py
"""
CropWindow V4 - נאמן לפונקציית MATLAB המקורית
שינויים מרכזיים:
1. סיבוב האובייקט כך שמישור החיתוך אופקי (Z קבוע)
2. סינון פאות לפי Z-center
3. מצלמה מאותחלת לתצוגה מיושרת (view 90,0)
4. תצוגת "תצוגה קטנה" של החלק שנחתך
"""
from __future__ import annotations
from artifact_app.gui.widgets import ArtifactButton
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QFrame, QSplitter
)


class CropWindow(QWidget):
    """חלון לחיתוך אובייקטים - נאמן ל-MATLAB"""

    mesh_applied = Signal(object)

    # הוספת positioning_mode לחתימה עם ברירת מחדל
    def __init__(self, mesh: pv.PolyData, parent=None, positioning_mode: str = "Normal Positioning"):
        super().__init__(parent)

        self.setWindowTitle("Crop Object Tool")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1200, 700)

        # שומרים את הסטטוס
        self.positioning_mode = positioning_mode

        # וידוא PolyData
        if not isinstance(mesh, pv.PolyData):
            try:
                mesh = mesh.extract_surface()
            except:
                pass

        # חישוב נורמלים
        try:
            mesh = mesh.compute_normals(
                point_normals=True,
                cell_normals=True,
                auto_orient_normals=True,
            )
        except Exception as e:
            print(f"Warning: normals: {e}")

        # שמירת מקור
        self._original_mesh = mesh.copy()
        self._current_mesh = mesh.copy()
        self._cut_mesh = None  # החלק שנחתך (להצגה)

        # מצב בחירה
        self._picking_active = False
        self._picked_points = []  # A, B, C
        self._marker_actors = []

        # פרמטרי חיתוך אחרון
        self._last_rotation_matrix = None
        self._last_cut_z = None
        self._last_cut_direction = None  # 'above' or 'below'

        self._is_closing = False

        self._init_ui()
        self._plot_mesh()
        self._set_default_camera()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- הוספת שורת אינדיקטור מצב בראש המסך ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #d0d7e2;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 5, 10, 5)

        lbl_title = QLabel("Crop Object Tool")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        lbl_mode = QLabel(f"[{self.positioning_mode}]")
        lbl_mode.setStyleSheet("font-weight: bold; font-size: 12px; color: #555;")
        lbl_mode.setAlignment(Qt.AlignCenter)

        # הוספת טקסט ריק בצד ימין רק כדי לשמור על איזון של flex (שהמצב ישב באמצע)
        lbl_spacer = QLabel("")

        header_layout.addWidget(lbl_title, 1)
        header_layout.addWidget(lbl_mode, 1)
        header_layout.addWidget(lbl_spacer, 1)

        layout.addWidget(header_frame, 0)
        # --- סוף הוספת שורת הכותרת ---

        # Splitter לתצוגה ראשית + תצוגה קטנה
        splitter = QSplitter(Qt.Horizontal)

        # --- תצוגה ראשית ---
        # (המשך הקוד נשאר אותו הדבר... main_frame, plotter, וכו')
        main_frame = QFrame()
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(main_frame)
        self.plotter.set_background("#e0e0e0")
        main_layout.addWidget(self.plotter.interactor, 1)

        splitter.addWidget(main_frame)

        # --- תצוגה קטנה (החלק שנחתך) ---
        # --- תצוגה קטנה (החלק שנחתך) ---
        side_frame = QFrame()
        side_frame.setObjectName("SidePanelRight")  # <--- הנה שורת הקסם!
        side_frame.setMaximumWidth(300)
        side_layout = QVBoxLayout(side_frame)
        side_layout.setContentsMargins(2, 2, 2, 2)

        self.lbl_cut_preview = QLabel("Cut Preview")
        self.lbl_cut_preview.setStyleSheet("font-weight: bold; color: #666;")
        self.lbl_cut_preview.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(self.lbl_cut_preview)

        self.plotter_small = QtInteractor(side_frame)
        self.plotter_small.set_background("#F0FAF9")
        side_layout.addWidget(self.plotter_small.interactor, 1)

        splitter.addWidget(side_frame)
        splitter.setSizes([900, 300])

        layout.addWidget(splitter, 1)

        # פאנל שליטה
        controls_frame = QFrame()
        # התיקון: תוחמים את העיצוב כך שישפיע רק על ה-QFrame עצמו
        controls_frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-top: 1px solid #ccc; }")
        controls_frame.setFixedHeight(65)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        self.lbl_instructions = QLabel("1. Rotate view  →  2. Click 'Start Cut'")
        self.lbl_instructions.setStyleSheet("font-weight: bold; color: #333; font-size: 13px;")

        self.btn_start = QPushButton("🔪 Start Cut")
        self.btn_start.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px;"
        )
        self.btn_start.clicked.connect(self._start_cut_mode)

        self.btn_invert = ArtifactButton("↔ Invert Side")
        self.btn_invert.clicked.connect(self._invert_clip)
        self.btn_invert.setEnabled(False)

        self.btn_reset = ArtifactButton("🔄 Reset")
        self.btn_reset.clicked.connect(self._reset_mesh)

        self.btn_view_cut = ArtifactButton("👁 Cut Normal View")
        self.btn_view_cut.clicked.connect(self._view_cut_normal)
        self.btn_view_cut.setEnabled(False)
        self.btn_apply = QPushButton("✓ Apply & Close")
        self.btn_apply.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px;"
        )
        self.btn_apply.clicked.connect(self._apply_and_close)

        controls_layout.addWidget(self.lbl_instructions, 1)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_invert)
        controls_layout.addWidget(self.btn_view_cut)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addWidget(self.btn_apply)

        layout.addWidget(controls_frame, 0)

    # ==================== הגדרות נראות - ללא שינוי ====================


    def _set_default_camera(self):
        """מצלמה התחלתית - תצוגת MC (חזית)"""
        if self._current_mesh is not None:
            center = self._current_mesh.center
            bounds = self._current_mesh.bounds
            dist = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2.5

            # מיקום המצלמה: מול האובייקט (בכיוון -Y)
            self.plotter.camera.position = (center[0], center[1] - dist, center[2])
            self.plotter.camera.focal_point = center
            self.plotter.camera.up = (0, 0, 1)  # Z למעלה

        self.plotter.reset_camera()
        self.plotter.reset_camera_clipping_range()

    def _plot_mesh(self, mesh=None, plotter=None):
        """ציור המודל עם המראה האחיד של Manual Positioning"""
        if mesh is None:
            mesh = self._current_mesh
        if plotter is None:
            plotter = self.plotter

        plotter.clear()

        if mesh is None or mesh.n_points == 0:
            return

        # רקע כחלחל-אפרפר
        #plotter.set_background("#DCE1E8")

        # תאורה אחידה
        plotter.enable_lightkit()

        # המודל עם אותן הגדרות חומר
        plotter.add_mesh(
            mesh,
            color="silver",
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.78,
            specular=0.20,
            specular_power=20,
            show_edges=False,
        )

        plotter.reset_camera()
        plotter.reset_camera_clipping_range()

    def _plot_cut_preview(self):
        """הצגת החלק שנחתך בחלון הקטן"""
        self.plotter_small.clear()

        if self._cut_mesh is None or self._cut_mesh.n_points == 0:
            self.lbl_cut_preview.setText("Cut Preview (empty)")
            return

        self.lbl_cut_preview.setText("Cut Preview (removed part)")

        # התיקון: שימוש בתאורה המובנית של PyVista
        self.plotter_small.enable_lightkit()

        # הנה כל החלק שנמחק בטעות - פרמטרי הציור וסגירת הפקודה:
        self.plotter_small.add_mesh(
            self._cut_mesh,
            color=(0.2, 1, 0.2),  # ירוק כמו ב-MATLAB
            smooth_shading=True,
            ambient=0.2,
            diffuse=0.7,
            specular=0.3,
            specular_power=30,
            show_edges=False,
        )

        self.plotter_small.reset_camera()
        self.plotter_small.reset_camera_clipping_range()
    # ==================== PICKING מתוקן ====================

    def _start_cut_mode(self):
        """התחלת בחירת נקודות"""
        print("=== START CUT MODE ===")
        self._picking_active = True
        self._picked_points = []
        self._remove_markers()

        self.lbl_instructions.setText("👆 Click POINT A (cut start) on object")
        self.lbl_instructions.setStyleSheet(
            "background-color: #FFF59D; color: #333; padding: 5px; "
            "font-weight: bold; border-radius: 3px;"
        )

        self.btn_start.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.btn_invert.setEnabled(False)
        self.btn_view_cut.setEnabled(False)

        # שימוש ב-callback ישיר על left click
        self.plotter.track_click_position(
            callback=self._on_point_picked,
            side='left',
            double=False,
        )
        print("track_click_position enabled")

    def _on_point_picked(self, point):
        """טיפול בבחירת נקודה"""
        print(f"_on_point_picked: {point}")

        if not self._picking_active:
            print("  not active")
            return

        if point is None:
            print("  point is None")
            return

        point = np.asarray(point).flatten()
        if point.size < 3 or np.any(np.isnan(point)):
            print("  invalid point")
            return

        # מצא נקודה קרובה על המש
        if self._current_mesh is not None and self._current_mesh.n_points > 0:
            idx = self._current_mesh.find_closest_point(point)
            closest = self._current_mesh.points[idx]
            dist = np.linalg.norm(point - closest)
            threshold = self._current_mesh.length * 0.2
            print(f"  closest on mesh: {closest}, dist: {dist:.3f}, thresh: {threshold:.3f}")
            if dist < threshold:
                point = closest

        self._picked_points.append(point.copy())
        self._add_marker(point)
        print(f"  total points: {len(self._picked_points)}")

        n = len(self._picked_points)

        if n == 1:
            self.lbl_instructions.setText("👆 Click POINT B (cut end) on object")
        elif n == 2:
            self.lbl_instructions.setText("👆 Click POINT C (side to REMOVE)")
        elif n >= 3:
            self._picking_active = False
            self._disable_picking()
            self._perform_cut_matlab_style()
            self.btn_start.setEnabled(True)
            self.btn_apply.setEnabled(True)
            self.lbl_instructions.setText("👆 Click POINT C (side to REMOVE)")
        elif n >= 3:
            self._picking_active = False
            self._disable_picking()
            self._perform_cut_matlab_style()
            self.btn_start.setEnabled(True)
            self.btn_apply.setEnabled(True)

    def _disable_picking(self):
        """כיבוי מצב picking"""
        print("_disable_picking called")
        try:
            self.plotter.untrack_click_position(side='left')
        except Exception as e:
            print(f"  untrack error: {e}")
        try:
            self.plotter.disable_picking()
        except Exception as e:
            print(f"  disable error: {e}")

    def _cancel_picking(self):
        self._picking_active = False
        self._picked_points = []
        self._remove_markers()
        self._disable_picking()

        self.lbl_instructions.setText("❌ Cancelled. Click 'Start Cut' to try again.")
        self.lbl_instructions.setStyleSheet("color: #c00; font-weight: bold;")

        self.btn_start.setEnabled(True)
        self.btn_apply.setEnabled(True)

    # ==================== שאר הפונקציות - ללא שינוי ====================

    def _add_marker(self, point):
        try:
            radius = self._current_mesh.length * 0.012
            sphere = pv.Sphere(radius=radius, center=point)

            n = len(self._picked_points)
            colors = ['red', 'blue', 'yellow']
            color = colors[min(n - 1, 2)]

            actor = self.plotter.add_mesh(
                sphere, color=color, lighting=False, reset_camera=False,
            )
            self._marker_actors.append(actor)
        except Exception as e:
            print(f"Marker error: {e}")

    def _remove_markers(self):
        for actor in self._marker_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self._marker_actors = []

    def _perform_cut_matlab_style(self):
        """
        חיתוך בסגנון MATLAB:
        1. חישוב שיפוע הקו A-B
        2. סיבוב האובייקט כך שמישור החיתוך אופקי
        3. סינון פאות לפי Z
        4. סיבוב חזרה
        """
        if len(self._picked_points) < 3:
            self._show_error("Need 3 points")
            return

        try:
            A = np.asarray(self._picked_points[0])
            B = np.asarray(self._picked_points[1])
            C = np.asarray(self._picked_points[2])

            # === שלב 1: חישוב מישור החיתוך ===
            line_vec = B - A
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-9:
                self._show_error("Points A and B are too close")
                return

            cam_dir = np.array(self.plotter.camera.direction)
            cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)

            plane_normal = np.cross(line_vec, cam_dir)
            norm = np.linalg.norm(plane_normal)

            if norm < 1e-6:
                self._show_error("Cut line parallel to view direction")
                self._reset_picking_state()
                return

            plane_normal = plane_normal / norm

            # === שלב 2: מציאת מטריצת סיבוב שהופכת את הנורמל ל-Z ===
            z_axis = np.array([0.0, 0.0, 1.0])

            v = np.cross(plane_normal, z_axis)
            c = np.dot(plane_normal, z_axis)

            if np.linalg.norm(v) < 1e-9:
                if c > 0:
                    R = np.eye(3)
                else:
                    R = np.diag([1, -1, -1])
            else:
                s = np.linalg.norm(v)
                v = v / s
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = np.eye(3) + vx * s + vx @ vx * (1 - c)

            # === שלב 3: סיבוב הנקודות והמש ===
            vertices = self._original_mesh.points.copy()
            center = vertices.mean(axis=0)

            vertices_centered = vertices - center
            vertices_rotated = (R @ vertices_centered.T).T + center

            A_rot = R @ (A - center) + center
            B_rot = R @ (B - center) + center
            C_rot = R @ (C - center) + center

            cut_z = (A_rot[2] + B_rot[2]) / 2.0

            if C_rot[2] >= cut_z:
                keep_below = True
                cut_direction = 'above'
            else:
                keep_below = False
                cut_direction = 'below'

            # === שלב 5: סינון פאות לפי Z ===
            faces = self._original_mesh.faces.reshape(-1, 4)[:, 1:4]

            kept_faces = []
            cut_faces = []

            for face in faces:
                face_verts = vertices_rotated[face]
                face_z_center = face_verts[:, 2].mean()

                if keep_below:
                    if face_z_center <= cut_z:
                        kept_faces.append(face)
                    else:
                        cut_faces.append(face)
                else:
                    if face_z_center >= cut_z:
                        kept_faces.append(face)
                    else:
                        cut_faces.append(face)

            if len(kept_faces) == 0:
                self._show_error("Cut removed entire object! Try inverting.")
                self._reset_picking_state()
                return

            # === שלב 6: סיבוב חזרה ===
            R_inv = R.T
            vertices_final = (R_inv @ (vertices_rotated - center).T).T + center

            kept_faces_arr = np.array(kept_faces)
            faces_with_count = np.column_stack([
                np.full(len(kept_faces_arr), 3),
                kept_faces_arr
            ]).flatten()

            new_mesh = pv.PolyData(vertices_final, faces_with_count)
            new_mesh = new_mesh.clean()

            if len(cut_faces) > 0:
                cut_faces_arr = np.array(cut_faces)
                cut_faces_with_count = np.column_stack([
                    np.full(len(cut_faces_arr), 3),
                    cut_faces_arr
                ]).flatten()
                self._cut_mesh = pv.PolyData(vertices_final, cut_faces_with_count)
                self._cut_mesh = self._cut_mesh.clean()
            else:
                self._cut_mesh = None

            try:
                new_mesh = new_mesh.compute_normals(auto_orient_normals=True)
                if self._cut_mesh is not None:
                    self._cut_mesh = self._cut_mesh.compute_normals(auto_orient_normals=True)
            except:
                pass

            self._last_rotation_matrix = R
            self._last_cut_z = cut_z
            self._last_cut_direction = cut_direction

            self._current_mesh = new_mesh
            self._plot_mesh()
            self._plot_cut_preview()
            self._remove_markers()

            self.lbl_instructions.setText("✓ Cut applied! Use 'Invert' if wrong side.")
            self.lbl_instructions.setStyleSheet(
                "background-color: #C8E6C9; color: #2E7D32; padding: 5px; "
                "font-weight: bold; border-radius: 3px;"
            )

            self.btn_invert.setEnabled(True)
            self.btn_view_cut.setEnabled(True)

        except Exception as e:
            self._show_error(f"Cut failed: {e}")
            import traceback
            traceback.print_exc()
            self._reset_picking_state()

    def _invert_clip(self):
        """היפוך - החלפה בין שני הצדדים"""
        if self._last_rotation_matrix is None:
            return

        if self._cut_mesh is not None and self._cut_mesh.n_points > 0:
            self._current_mesh, self._cut_mesh = self._cut_mesh, self._current_mesh
            self._plot_mesh()
            self._plot_cut_preview()
            self.lbl_instructions.setText("✓ Cut inverted!")

    def _view_cut_normal(self):
        """מבט ניצב למישור החיתוך"""
        if self._last_rotation_matrix is None:
            return

        R = self._last_rotation_matrix
        cut_normal = R.T @ np.array([0, 0, 1])

        center = self._current_mesh.center
        focal = center
        position = center + cut_normal * self._current_mesh.length * 2

        self.plotter.camera.position = position
        self.plotter.camera.focal_point = focal
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.reset_camera()
        self.plotter.reset_camera_clipping_range()

    def _reset_mesh(self):
        self._current_mesh = self._original_mesh.copy()
        self._cut_mesh = None
        self._plot_mesh()
        self.plotter_small.clear()
        self.lbl_cut_preview.setText("Cut Preview")
        self._reset_picking_state()

        self.lbl_instructions.setText("🔄 Reset. Click 'Start Cut' to begin.")
        self.lbl_instructions.setStyleSheet("color: #333; font-weight: bold;")

        self._last_rotation_matrix = None
        self._last_cut_z = None
        self._last_cut_direction = None
        self.btn_invert.setEnabled(False)
        self.btn_view_cut.setEnabled(False)

    def _reset_picking_state(self):
        self._picking_active = False
        self._picked_points = []
        self._remove_markers()
        self._disable_picking()
        self.btn_start.setEnabled(True)
        self.btn_apply.setEnabled(True)

    def _show_error(self, msg: str):
        self.lbl_instructions.setText(f"❌ {msg}")
        self.lbl_instructions.setStyleSheet(
            "background-color: #FFCDD2; color: #C62828; padding: 5px; "
            "font-weight: bold; border-radius: 3px;"
        )

    def _apply_and_close(self):
        if self._current_mesh is not None and self._current_mesh.n_points > 0:
            try:
                self._current_mesh = self._current_mesh.compute_normals(
                    auto_orient_normals=True
                )
            except:
                pass
            self.mesh_applied.emit(self._current_mesh)

        self._is_closing = True
        self.close()

    def closeEvent(self, event):
        self._is_closing = True
        self._disable_picking()

        try:
            self.plotter.clear()
            self.plotter.close()
        except:
            pass

        try:
            self.plotter_small.clear()
            self.plotter_small.close()
        except:
            pass

        self._original_mesh = None
        self._current_mesh = None
        self._cut_mesh = None
        self._marker_actors = []

        super().closeEvent(event)