import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame, QFormLayout,
    QComboBox, QFileDialog, QMessageBox, QDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import time


class AngleBetweenSurfacesWindow(QMainWindow):
    # הוספנו את positioning_mode לחתימה
    def __init__(self, mesh: pv.PolyData, parent=None, positioning_mode: str = "Normal Positioning"):
        super().__init__(parent)
        self.setWindowTitle("Calculate Angle Between Surfaces (Multi-Point Path)")
        self.resize(1100, 750)

        self.positioning_mode = positioning_mode  # שמירת המצב

        if not isinstance(mesh, pv.PolyData):
            try:
                mesh = mesh.extract_surface()
            except:
                pass

        try:
            mesh = mesh.compute_normals(
                point_normals=True, cell_normals=True, auto_orient_normals=True,
            )
        except Exception as e:
            print(f"Warning: normals: {e}")

        self.mesh = mesh
        self._current_mesh = self.mesh.copy()

        self._picking_active = False
        self._picked_points = []
        self._draw_points = []  # רשימה נפרדת לנקודות התצוגה המרחפות
        self._surface_actors = []

        self._last_calc_data_list = []
        self._measurements_list = []
        self._last_click_time = 0

        self._init_ui()
        self._plot_mesh()
        self._set_default_camera()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # הפכנו את הלייאאוט הראשי לאנכי (VBox) כדי לשים כותרת למעלה ותוכן למטה
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- יצירת שורת הכותרת (Header) ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #d0d7e2; border-bottom: 1px solid #ccc;")
        header_frame.setFixedHeight(40)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 0, 15, 0)

        lbl_title = QLabel("Angle Between Surfaces")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")

        lbl_mode = QLabel(f"[{self.positioning_mode}]")
        lbl_mode.setStyleSheet("font-weight: bold; font-size: 12px; color: #555; border: none;")
        lbl_mode.setAlignment(Qt.AlignCenter)

        lbl_spacer = QLabel("")  # לאיזון הפריסה

        header_layout.addWidget(lbl_title, 1)
        header_layout.addWidget(lbl_mode, 1)
        header_layout.addWidget(lbl_spacer, 1)

        main_layout.addWidget(header_frame, 0)

        # --- יצירת אזור התוכן המפוצל (פאנל שמאלי ומודל תלת-ממד) ---
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        left_panel = QFrame()
        left_panel.setFixedWidth(260)
        left_layout = QVBoxLayout(left_panel)

        # --- אזור הגדרות בסיסיות ---
        basic_form_layout = QFormLayout()

        self.input_h = QLineEdit()
        self.input_h.setPlaceholderText("e.g. 5.0")
        basic_form_layout.addRow("Analysis Width (mm)", self.input_h)

        self.input_method = QComboBox()
        self.input_method.addItems(["Legacy (Mean Surfaces)", "New (Mean Angle)"])
        basic_form_layout.addRow("Method:", self.input_method)

        left_layout.addLayout(basic_form_layout)

        # --- כפתור פתיחה/סגירה ל-Advanced ---
        self.btn_advanced = QPushButton("Advanced Options ▼")
        self.btn_advanced.setCheckable(True)  # מאפשר לכפתור לשמור מצב (לחוץ/לא לחוץ)
        self.btn_advanced.setStyleSheet(
            "text-align: left; font-weight: bold; color: #555; border: none; padding-top: 10px; padding-bottom: 5px;")
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        left_layout.addWidget(self.btn_advanced)

        # --- אזור Advanced מוסתר ---
        self.advanced_widget = QWidget()
        advanced_form = QFormLayout(self.advanced_widget)
        advanced_form.setContentsMargins(15, 0, 0, 10)  # הזחה קלה פנימה כדי להראות שזה תת-תפריט

        self.input_dilute = QLineEdit()
        recommended_dilute = max(1, self.mesh.n_cells // 10000)
        self.input_dilute.setText(str(recommended_dilute))
        advanced_form.addRow("Dilute Factor:", self.input_dilute)

        self.advanced_widget.hide()  # מוסתר כברירת מחדל
        left_layout.addWidget(self.advanced_widget)

        # ... (כאן ממשיך הקוד של כפתור btn_select_points) ...
        self.btn_select_points = QPushButton("1. Start Path (Click Points)")
        self.btn_select_points.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_select_points.clicked.connect(self._start_picking_mode)
        left_layout.addWidget(self.btn_select_points)

        self.btn_clear_path = QPushButton("🔄 Clear / Restart Path")
        self.btn_clear_path.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px;")
        self.btn_clear_path.clicked.connect(self._clear_path)
        self.btn_clear_path.hide()
        left_layout.addWidget(self.btn_clear_path)

        self.btn_calculate = QPushButton("2. Finish & Calculate")
        self.btn_calculate.setStyleSheet("background-color: #001F3F; color: white; font-weight: bold; padding: 8px;")
        self.btn_calculate.setEnabled(False)
        self.btn_calculate.clicked.connect(self._calculate_angle)
        left_layout.addWidget(self.btn_calculate)

        self.lbl_result = QLabel("Result: ---")
        self.lbl_result.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 15px;")
        left_layout.addWidget(self.lbl_result)

        self.export_group = QFrame()
        export_v = QVBoxLayout(self.export_group)
        export_v.setContentsMargins(0, 20, 0, 0)

        self.btn_save_to_list = QPushButton("💾 Add Path to List")
        self.btn_save_to_list.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save_to_list.setEnabled(False)
        self.btn_save_to_list.clicked.connect(self._save_to_list)
        export_v.addWidget(self.btn_save_to_list)

        self.lbl_saved_count = QLabel("Saved segments: 0")
        self.lbl_saved_count.setStyleSheet("color: #555; font-style: italic; margin-bottom: 15px;")
        export_v.addWidget(self.lbl_saved_count)

        self.btn_show_plot = QPushButton("📈 Show Graph")
        self.btn_show_plot.setEnabled(False)
        self.btn_show_plot.clicked.connect(self._show_plot)
        self.btn_show_plot.hide()
        export_v.addWidget(self.btn_show_plot)

        self.btn_help = QPushButton("ℹ️ Help / Guide")
        self.btn_help.setStyleSheet("background-color: #B0BEC5; color: black; font-weight: bold; padding: 6px;")
        self.btn_help.clicked.connect(self._show_help_dialog)
        export_v.addWidget(self.btn_help)

        self.btn_save_img = QPushButton("📸 Save Image")
        self.btn_save_img.clicked.connect(self._save_image)
        export_v.addWidget(self.btn_save_img)

        self.btn_export_excel = QPushButton("📊 Export All to Excel")
        self.btn_export_excel.setEnabled(False)
        self.btn_export_excel.clicked.connect(self._export_to_excel)
        export_v.addWidget(self.btn_export_excel)

        left_layout.addWidget(self.export_group)
        left_layout.addStretch()

        # חיבור הפאנל השמאלי לאזור התוכן (במקום ל-main_layout הישן)
        content_layout.addWidget(left_panel)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#F0F0F0")

        # חיבור פלוטר התלת-ממד לאזור התוכן (במקום ל-main_layout הישן)
        content_layout.addWidget(self.plotter.interactor)

    def _toggle_advanced(self, checked):
        """מציג או מסתיר את תפריט ההגדרות המתקדמות"""
        if checked:
            self.advanced_widget.show()
            self.btn_advanced.setText("Advanced Options ▲")
        else:
            self.advanced_widget.hide()
            self.btn_advanced.setText("Advanced Options ▼")

    def _setup_lighting(self, plotter):
        plotter.enable_lightkit()
        try:
            plotter.enable_anti_aliasing()
        except:
            pass

    def _set_default_camera(self):
        if self._current_mesh is not None:
            center = self._current_mesh.center
            bounds = self._current_mesh.bounds
            dist = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2.5
            self.plotter.camera.position = (center[0], center[1] - dist, center[2])
            self.plotter.camera.focal_point = center
            self.plotter.camera.up = (0, 0, 1)
        self.plotter.reset_camera()

    def _plot_mesh(self):
        self.plotter.clear()
        self._setup_lighting(self.plotter)
        # שומרים את האקטור של המודל כדי שהעכבר יינעל אך ורק עליו
        self._mesh_actor = self.plotter.add_mesh(
            self._current_mesh,
            color="#AAAAAA",
            smooth_shading=True,
            show_edges=False
        )

    def _start_picking_mode(self):
        self._picking_active = True
        self._picked_points = []
        self._draw_points = []
        self._remove_markers()

        self._update_picking_button()
        self.btn_clear_path.show()
        self.btn_calculate.setEnabled(False)

        self.plotter.interactor.setCursor(Qt.CrossCursor)

        # חיבור מנגנון ה-VTK המדויק
        if not hasattr(self, '_pick_observer_id'):
            self._pick_observer_id = self.plotter.iren.add_observer("LeftButtonPressEvent", self._perform_pick)

    def _clear_path(self):
        self._picked_points = []
        self._draw_points = []
        self._remove_markers()
        self._update_picking_button()
        self.btn_calculate.setEnabled(False)
        self.btn_calculate.setStyleSheet("background-color: #001F3F; color: white; font-weight: bold; padding: 8px;")

    def _update_picking_button(self):
        count = len(self._picked_points)
        self.btn_select_points.setText(f"📍 Picking... [ {count} pts ] (Click Finish)")
        self.btn_select_points.setStyleSheet("background-color: #FFF59D; font-weight: bold;")

    def _perform_pick(self, obj, event):
        """פונקציית צלף VTK - פוגעת רק במודל ומתעלמת מכל השאר"""
        if not self._picking_active: return

        current_time = time.time()
        if current_time - self._last_click_time < 0.15: return
        self._last_click_time = current_time

        click_pos = self.plotter.iren.get_event_position()

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        # נעילת הלחיצה אך ורק למודל הראשי (מונע לחיצות על צינורות המסלול בטעות)
        if hasattr(self, '_mesh_actor'):
            picker.AddPickList(self._mesh_actor)
            picker.PickFromListOn()

        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)

        if picker.GetCellId() != -1:
            pick_pos = np.array(picker.GetPickPosition())
            self._handle_picked_point(pick_pos)

    def _handle_picked_point(self, point):
        self._picked_points.append(point.copy())

        # משיכת הנקודה המצוירת מחוץ למודל כדי למנוע היבלעות
        try:
            idx = self._current_mesh.find_closest_point(point)
            normal = self._current_mesh.point_normals[idx]
            # דחיפה קלה החוצה (0.4% מאורך המודל)
            draw_point = point + normal * (self._current_mesh.length * 0.004)
        except:
            draw_point = point

        self._draw_points.append(draw_point)
        self._update_picking_button()

        points_array = np.vstack(self._draw_points)
        count = len(self._draw_points)

        # 1. ציור הנקודה הראשונה (A) ככדור כחול בולט
        blue_point = pv.PolyData(points_array[0:1])
        self.plotter.add_mesh(
            blue_point, color='blue', point_size=20,
            render_points_as_spheres=True, name='first_point_actor', pickable=False
        )

        # 2. ציור שאר הנקודות והצינור הצהוב
        if count > 1:
            # שאר הנקודות במסלול ככדורים אדומים
            red_points = pv.PolyData(points_array[1:])
            self.plotter.add_mesh(
                red_points, color='red', point_size=20,
                render_points_as_spheres=True, name='rest_points_actor', pickable=False
            )

            # בניית הקו (צינור צהוב)
            lines = np.empty((count - 1, 3), dtype=int)
            lines[:, 0] = 2
            lines[:, 1] = np.arange(count - 1)
            lines[:, 2] = np.arange(1, count)

            lines_poly = pv.PolyData(points_array, lines=lines)
            self.plotter.add_mesh(
                lines_poly, color='yellow', line_width=6,
                render_lines_as_tubes=True, name='picked_lines_actor', pickable=False
            )

            self.btn_calculate.setEnabled(True)
            self.btn_calculate.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def _remove_markers(self):
        # מחיקת כל שכבות המסלול השונות
        self.plotter.remove_actor('first_point_actor')
        self.plotter.remove_actor('rest_points_actor')
        self.plotter.remove_actor('picked_lines_actor')

        if hasattr(self, '_surface_actors'):
            for actor in self._surface_actors: self.plotter.remove_actor(actor)
        self._surface_actors = []
    def _calculate_angle(self):
        self._picking_active = False
        self.plotter.interactor.setCursor(Qt.ArrowCursor)

        self.btn_clear_path.hide()
        self.btn_select_points.setText("1. Start Path (Click Points)")
        self.btn_select_points.setStyleSheet("font-weight: bold; padding: 8px;")

        if len(self._picked_points) < 2:
            return

        try:
            max_h = float(self.input_h.text().strip())
            dil = int(self.input_dilute.text())
        except:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers.")
            return

        method = self.input_method.currentText()
        h_array = np.linspace(0, max_h, 21)
        shell_thick = max_h / 4.0

        centers = self.mesh.cell_centers().points[::dil]

        # 1. חישוב השטחים של כל הפאות (המשולשים) במודל
        mesh_with_areas = self.mesh.compute_cell_sizes()
        areas = mesh_with_areas.cell_data["Area"]

        # 2. שקלול הנורמלים: הכפלת וקטור היחידה בשטח כפול 2 (כדי לדמות מכפלה וקטורית של צלעות)
        weighted_normals = self.mesh.cell_normals * (areas[:, np.newaxis] * 2)

        # 3. הפעלת הדילול (Dilute) על הנורמלים המשוקללים
        normals = weighted_normals[::dil]

        self._last_calc_data_list = []
        all_final_angles = []
        all_final_sds = []

        if hasattr(self, '_surface_actors'):
            for actor in self._surface_actors: self.plotter.remove_actor(actor)
        self._surface_actors = []

        # חישוב האלגוריתם (משתמש בנקודות האמיתיות שעל המשטח, לא בנקודות הציפה)
        for seg_idx in range(len(self._picked_points) - 1):
            A = np.array(self._picked_points[seg_idx])
            B = np.array(self._picked_points[seg_idx + 1])
            d_len = np.linalg.norm(B - A)
            t = (B - A) / d_len

            r2A = centers - A
            dot_r2A_t = np.dot(r2A, t)
            dist_sq = np.sum(r2A ** 2, axis=1) - dot_r2A_t ** 2

            valid_len_mask = (dot_r2A_t >= 0) & (dot_r2A_t <= d_len)
            v_centers, v_normals, v_dist_sq = centers[valid_len_mask], normals[valid_len_mask], dist_sq[valid_len_mask]

            results = []
            pts_left_list = []
            pts_right_list = []

            for th in range(21):
                mask = (v_dist_sq >= h_array[th] ** 2) & (v_dist_sq <= (h_array[th] + shell_thick) ** 2)
                sn, sr = v_normals[mask], v_centers[mask]

                if len(sn) < 2:
                    results.append([np.nan, np.nan])
                    pts_left_list.append(np.array([]))
                    pts_right_list.append(np.array([]))
                    continue

                mid_plane = np.cross(t, np.sum(sn, axis=0))
                l_mask = np.dot(sn, mid_plane) > 0
                ln, rn = sn[l_mask], sn[~l_mask]
                lr, rr = sr[l_mask], sr[~l_mask]

                pts_left_list.append(lr)
                pts_right_list.append(rr)

                if len(ln) == 0 or len(rn) == 0:
                    results.append([np.nan, np.nan])
                    continue

                if "Legacy" in method:
                    nl, nr = np.sum(ln, 0), np.sum(rn, 0)
                    ang = np.degrees(
                        np.pi - np.arccos(np.clip(np.dot(nl, nr) / (np.linalg.norm(nl) * np.linalg.norm(nr)), -1, 1)))
                    results.append([ang, 0])
                else:
                    ln_u, rn_u = ln / np.linalg.norm(ln, axis=1, keepdims=True), rn / np.linalg.norm(rn, axis=1,
                                                                                                     keepdims=True)
                    alphas = 180.0 - np.degrees(np.arccos(np.clip(np.dot(ln_u, rn_u.T), -1, 1)))
                    results.append([np.mean(alphas), np.std(alphas)])

            results = np.array(results)
            q_vals = np.full(16, np.nan)
            for i in range(16):
                win = results[i:i + 6, 0]
                if not np.any(np.isnan(win)):
                    q_vals[i] = np.sqrt(np.sum(np.diff(win) ** 2) / 5.0)

            if np.all(np.isnan(q_vals)):
                continue

            best_q = np.nanargmin(q_vals)
            final_ang = np.mean(results[best_q:best_q + 6, 0])
            final_sd = np.mean(results[best_q:best_q + 6, 1])

            all_final_angles.append(final_ang)
            all_final_sds.append(final_sd)

            self._last_calc_data_list.append({
                "summary": {"angle": final_ang, "sd": final_sd},
                "h_test": {"h1": h_array, "angles": results[:, 0],
                           "quality": np.pad(q_vals, (0, 5), constant_values=np.nan)},
                "segment": {"len": d_len, "A": A, "B": B},
                "best_q": best_q
            })

            best_left_points = []
            best_right_points = []
            for i in range(best_q, best_q + 6):
                if len(pts_left_list[i]) > 0: best_left_points.append(pts_left_list[i])
                if len(pts_right_list[i]) > 0: best_right_points.append(pts_right_list[i])

            if len(best_left_points) > 0:
                a_l = self.plotter.add_mesh(pv.PolyData(np.vstack(best_left_points)), color='blue', point_size=5,
                                            style='points')
                self._surface_actors.append(a_l)

            if len(best_right_points) > 0:
                a_r = self.plotter.add_mesh(pv.PolyData(np.vstack(best_right_points)), color='red', point_size=5,
                                            style='points')
                self._surface_actors.append(a_r)

        if not all_final_angles:
            self.lbl_result.setText("Error: Unstable path.")
            return

        global_ang = np.mean(all_final_angles)
        global_sd = np.mean(all_final_sds)
        self.lbl_result.setText(
            f"Path Mean ({len(all_final_angles)} segs):\n{global_ang:.2f}°\nMean SD: {global_sd:.2f}")

        self.btn_save_to_list.setEnabled(True)
        if len(self._last_calc_data_list) == 1:
            self.btn_show_plot.setEnabled(True)
        else:
            self.btn_show_plot.setEnabled(False)

    def _save_to_list(self):
        if not self._last_calc_data_list: return

        self._measurements_list.extend(self._last_calc_data_list)

        count = len(self._measurements_list)
        self.lbl_saved_count.setText(f"Saved segments: {count}")
        self.btn_save_to_list.setEnabled(False)
        self.btn_export_excel.setEnabled(True)

    def _show_plot(self):
        if not self._last_calc_data_list or len(self._last_calc_data_list) > 1: return

        d = self._last_calc_data_list[0]
        h_vals = d["h_test"]["h1"]
        angles = d["h_test"]["angles"]
        best_q = d["best_q"]

        dialog = QDialog(self)
        dialog.setWindowTitle("Angle vs. Distance (h) Plot")
        dialog.resize(600, 450)
        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.plot(h_vals, angles, 'b-o', label='Angle at h', alpha=0.6)

        if best_q is not None and not np.isnan(best_q):
            start_h = h_vals[best_q]
            end_h = h_vals[best_q + 5]
            ax.axvspan(start_h, end_h, color='green', alpha=0.15, label='Stable Region')
            ax.plot(h_vals[best_q:best_q + 6], angles[best_q:best_q + 6], 'ro')

        ax.set_xlabel("Distance from edge (h) [mm]")
        ax.set_ylabel("Mean Angle (degrees)")
        ax.set_title("h_test Analysis")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        dialog.exec()

    def _save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg)")
        if path:
            self.plotter.screenshot(path)
            QMessageBox.information(self, "Success", f"Image saved to:\n{path}")

    def _export_to_excel(self):
        if not self._measurements_list: return

        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "", "Excel Files (*.xlsx)")
        if not path: return

        summary_rows = []
        segments_rows = []

        for i, d in enumerate(self._measurements_list):
            meas_id = i + 1

            segments_rows.append({
                'Segment #': meas_id,
                'mean angle': d["summary"]["angle"],
                'SD': d["summary"]["sd"],
                'segment length': d["segment"]["len"],
                'z start': d["segment"]["A"][2], 'y start': d["segment"]["A"][1], 'x start': d["segment"]["A"][0],
                'z end': d["segment"]["B"][2], 'y end': d["segment"]["B"][1], 'x end': d["segment"]["B"][0]
            })

            for step in range(21):
                row = {
                    'Segment #': meas_id if step == 0 else '',
                    'mean angle measurements': d["summary"]["angle"] if step == 0 else np.nan,
                    'SD': d["summary"]["sd"] if step == 0 else np.nan,
                    'h1': d["h_test"]["h1"][step],
                    'angle(h1)': d["h_test"]["angles"][step],
                    'quality(h1)': d["h_test"]["quality"][step]
                }
                summary_rows.append(row)

            summary_rows.append({k: np.nan for k in summary_rows[-1].keys()})

        # --- התוספת החדשה: שורת סיכום כולל ---
        if segments_rows:
            all_angles = [r['mean angle'] for r in segments_rows]
            all_sds = [r['SD'] for r in segments_rows]

            # שורת רווח ליופי ולהפרדה ברורה
            segments_rows.append({k: np.nan for k in segments_rows[0].keys()})

            # הוספת הממוצע הכללי
            segments_rows.append({
                'Segment #': 'Grand Mean',
                'mean angle': np.mean(all_angles),
                'SD': np.mean(all_sds),
                'segment length': np.nan,
                'z start': np.nan, 'y start': np.nan, 'x start': np.nan,
                'z end': np.nan, 'y end': np.nan, 'x end': np.nan
            })

        df_summary = pd.DataFrame(summary_rows)
        df_segments = pd.DataFrame(segments_rows)

        with pd.ExcelWriter(path) as writer:
            df_summary.to_excel(writer, sheet_name='h_test_data', index=False)
            df_segments.to_excel(writer, sheet_name='segments_summary', index=False)

        QMessageBox.information(self, "Success", f"Exported {len(self._measurements_list)} segments to:\n{path}")
    def _show_help_dialog(self):
        help_text = """<h3>User Guide - Angle Measurement (h_test)</h3>

        <b>What this tool does:</b><br>
        This tool objectively measures the angle of a ridge or edge on an artifact. <br>
        It performs a "depth scan" (h_test), sampling the surface at 21 increasing distances from the edge to find the most "stable" angle, ignoring micro-fractures and body curvature.<br><br>

        <b>Calculation Methods:</b><br>
        <ul>
            <li><b>Legacy (Mean Surfaces):</b> Averages all surface normals from the right and left sides into a single plane for each side. Less sensitive to noise.</li>
            <li><b>New (Mean Angle):</b> Calculates the angle between <i>every</i> pair of opposing triangles, outputting a Mean Angle and SD. The SD indicates how "noisy" or weathered the surface is.</li>
        </ul>

        <b>How to Use (Single or Multi-Point Path):</b><br>
        1. <b>Start Path:</b> Click 'Start Path' and mark points along the ridge. You can click 2 points for a simple line, or multiple points along a curved edge. A yellow line will trace your path.<br>
        2. <b>Clear Path:</b> If you made a mistake, click 'Clear / Restart Path' to erase the current selection.<br>
        3. <b>Calculate:</b> Click 'Finish & Calculate'. The software divides the path into segments, calculates the h_test for each, and displays the global mean angle.<br>
        4. <b>Save & Export:</b> Click 'Add Path to List' to store all segments in memory. When finished with the artifact, click 'Export All to Excel'.
        """
        QMessageBox.information(self, "Help / Guide", help_text)

    def closeEvent(self, event):
        try:
            self.plotter.close()
        except:
            pass
        super().closeEvent(event)