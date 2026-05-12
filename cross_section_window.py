# -*- coding: utf-8 -*-
# src/artifact_app/gui/cross_section_window.py
from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

# מאפשר התנהגות גמישה יותר של PyVista (קריטי לצינורות שלנו)
pv.global_theme.allow_empty_mesh = True

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QDialog,
    QFileDialog, QSizePolicy, QMenu, QLayout, QMessageBox, QFrame
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from artifact_app.gui.live_five_views_widget import LiveFiveViewsWidget
from artifact_app.processing.hatach_engine import compute_hatach_contours


# ============================================================
# Helpers: יצירת אובייקטים דו-ממדיים חסינים
# ============================================================
def _create_overlay_line(pt1_3d, pt2_3d, color, line_width=3):
    points = vtk.vtkPoints()
    points.InsertNextPoint(pt1_3d[0], pt1_3d[1], pt1_3d[2])
    points.InsertNextPoint(pt2_3d[0], pt2_3d[1], pt2_3d[2])

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)

    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coord)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.PickableOff()

    c_map = {'green': (0.0, 0.8, 0.0), 'red': (1.0, 0.0, 0.0), 'black': (0.0, 0.0, 0.0)}
    c = c_map.get(color, (0, 0, 0))
    actor.GetProperty().SetColor(c[0], c[1], c[2])
    actor.GetProperty().SetLineWidth(line_width)

    return actor


# ============================================================
# חלון התוצאה הדו-ממדי (process_cut)
# ============================================================
# במקום class Result2DWindow(QDialog):
class Result2DWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D Cut Result")
        self.resize(700, 700)

        # דגלים לחלון רגיל עם כפתורי שליטה
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # הבטחה שהחלון לא מודאלי
        self.setWindowModality(Qt.NonModal)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # ... (המשך הקוד שלך עם self.figure וכו')

        # --- (מכאן והלאה הקוד שלך נשאר זהה לחלוטין) ---
        self.figure = Figure(figsize=(5, 5), dpi=100)
        # רקע לבן חלק סביב הקנבס
        self.figure.patch.set_facecolor('white')
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.canvas.mpl_connect('button_press_event', self.parent()._on_2d_click)

        layout.addWidget(self.canvas)

        self.is_looking_up = False

        bottom_layout = QHBoxLayout()
        self.lbl_instructions = QLabel("Right-click to measure | Left-click to lock/clear")
        bottom_layout.addWidget(self.lbl_instructions)

        bottom_layout.addStretch()

        self.btn_flip_view = QPushButton("View: Looking DOWN")
        self.btn_flip_view.setStyleSheet(
            "background-color: #001F3F; color: white; font-weight: bold; padding: 6px 15px;")
        self.btn_flip_view.clicked.connect(self.on_flip_view_clicked)
        bottom_layout.addWidget(self.btn_flip_view)

        self.btn_save = QPushButton("Save Contour (TXT/CSV)")
        self.btn_save.setStyleSheet("background-color: #001F3F; color: white; font-weight: bold; padding: 6px 15px;")
        self.btn_save.clicked.connect(self.on_save_clicked)
        bottom_layout.addWidget(self.btn_save)

        self.btn_save_img = QPushButton("Save Image (PNG/JPG)")
        self.btn_save_img.setStyleSheet(
            "background-color: #001F3F; color: white; font-weight: bold; padding: 6px 15px;")
        self.btn_save_img.clicked.connect(self.on_save_image_clicked)
        bottom_layout.addWidget(self.btn_save_img)

        self.btn_save_excel = QPushButton("Save to Excel (Data + Image)")
        self.btn_save_excel.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 15px;")
        self.btn_save_excel.clicked.connect(self.on_save_excel_clicked)
        bottom_layout.addWidget(self.btn_save_excel)

        layout.addLayout(bottom_layout)
        self.current_contours = []

    def on_flip_view_clicked(self):
        self.is_looking_up = not self.is_looking_up
        if self.is_looking_up:
            self.btn_flip_view.setText("View: Looking UP")
        else:
            self.btn_flip_view.setText("View: Looking DOWN")
        self.parent().redraw_2d_view()

        if hasattr(self.parent(), 'cut_confirmed_signal'):
            pixmap = self.canvas.grab()
            # הוספנו את הפרמטר הרביעי self.parent().current_z כדי שהאיתות יעבוד
            self.parent().cut_confirmed_signal.emit(
                pixmap,
                self.parent().current_contours_data,
                self.is_looking_up,
                self.parent().current_z
            )
    def set_contours(self, contours):
        self.current_contours = contours

    def on_save_clicked(self):
        if not self.current_contours:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Contour Data", "",
                                                   "CSV Files (*.csv);;Text Files (*.txt)")
        if not file_path:
            return

        main_contour = max(self.current_contours, key=len)
        x = main_contour[:, 0]
        y = main_contour[:, 1]
        signed_area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1])

        if signed_area > 0:
            main_contour = main_contour[::-1]

        np.savetxt(file_path, main_contour, delimiter=',', header='X,Y', comments='', fmt='%.5f')
        print(f"Successfully saved clockwise contour to {file_path}")

    def on_save_image_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure Image", "",
                                                   "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf)")
        if not file_path:
            return

        self.figure.savefig(file_path, bbox_inches='tight', dpi=300)
        print(f"Successfully saved figure image to {file_path}")

    def on_save_excel_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel Report", "", "Excel Files (*.xlsx)")
        if not file_path:
            return

        import pandas as pd
        import tempfile
        import os
        from PySide6.QtWidgets import QMessageBox

        # 1. איסוף המדידות מהחלון הראשי (האבא)
        measurements = self.parent().saved_measurements
        data = []

        for i, meas in enumerate(measurements, 1):
            pts = meas['points']
            if meas['type'] == 'dist':
                d = np.linalg.norm(pts[1] - pts[0])
                data.append({
                    "Measurement #": i,
                    "Type": "Distance",
                    "Value": f"{d:.2f} mm",
                    "Details": f"From ({pts[0][0]:.1f}, {pts[0][1]:.1f}) to ({pts[1][0]:.1f}, {pts[1][1]:.1f})"
                })
            elif meas['type'] == 'angle':
                ba = pts[0] - pts[1]
                bc = pts[2] - pts[1]
                ang = np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1)))
                data.append({
                    "Measurement #": i,
                    "Type": "Angle",
                    "Value": f"{ang:.1f}°",
                    "Details": f"Vertex at ({pts[1][0]:.1f}, {pts[1][1]:.1f})"
                })

        if not data:
            data.append({"Measurement #": "-", "Type": "No manual measurements", "Value": "-", "Details": "-"})

        df = pd.DataFrame(data)

        try:
            # 2. ייצוא הטבלה לאקסל
            df.to_excel(file_path, index=False, sheet_name="Cut Analysis")

            # 3. שמירת תמונת הגרף לקובץ זמני
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_png = tmp.name

            # שמירה ברזולוציה סבירה לאקסל (150 dpi)
            self.figure.savefig(temp_png, bbox_inches='tight', dpi=150)

            # 4. הזרקת התמונה לאקסל בעזרת openpyxl
            import openpyxl
            from openpyxl.drawing.image import Image as ExcelImage

            wb = openpyxl.load_workbook(file_path)
            ws = wb["Cut Analysis"]

            # עיצוב רוחב עמודות בסיסי
            ws.column_dimensions['A'].width = 15
            ws.column_dimensions['B'].width = 15
            ws.column_dimensions['C'].width = 15
            ws.column_dimensions['D'].width = 45

            # הוספת התמונה בתא F2
            img = ExcelImage(temp_png)

            # --- שמירה על פרופורציות מושלמות ---
            target_width = 350  # הרוחב שאנחנו רוצים באקסל

            # חישוב היחס המקורי (גובה חלקי רוחב)
            aspect_ratio = img.height / img.width

            # קביעת הגדלים החדשים
            img.width = target_width
            img.height = int(target_width * aspect_ratio)
            # ------------------------------------

            ws.add_image(img, "F2")

            wb.save(file_path)
            wb.close()

            # מחיקת הקובץ הזמני
            os.remove(temp_png)

            print(f"Successfully saved Excel to {file_path}")
            QMessageBox.information(self, "Success", f"Report saved successfully:\n{file_path}")

        except ImportError:
            QMessageBox.critical(self, "Missing Library",
                                 "Please install openpyxl to export images to Excel:\npip install openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save Excel file:\n{e}")

    def closeEvent(self, event):
        self.hide()
        event.ignore()


# ============================================================
# החלון הראשי
# ============================================================
class CrossSectionWindow(QMainWindow):
    # איתות שידור תמונה
    cut_confirmed_signal = Signal(object, object, bool, float)

    # הנה התיקון הקריטי! measurements חזר למקומו הטבעי
    # הוספנו positioning_mode לחתימה
    def __init__(self, mesh, measurements=None, title="Cross Section", filename="", parent=None,
                 positioning_mode="Normal Positioning"):
        super().__init__(parent)
        self.setWindowTitle(title if not filename else f"Cross Section: {filename}")
        self.resize(1400, 900)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.mesh = mesh
        self.measurements = measurements
        self.filename = filename
        self.positioning_mode = positioning_mode  # שמירת המצב במחלקה

        # (מכאן והלאה הקוד נשאר אותו דבר: bounds = self.mesh.bounds וכו'...)
        bounds = self.mesh.bounds
        self.z_min, self.z_max = bounds[4], bounds[5]
        self.current_z = (self.z_min + self.z_max) / 2.0

        self.actors_cache = {}
        self.measure_points = []
        self.current_contours_data = []
        self.saved_measurements = []

        self.cut_mesh = pv.PolyData()
        self._is_slicing = False
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(lambda: self.update_cut_line(force=True))

        self.result_window = Result2DWindow(self)

        self._init_ui()
        QTimer.singleShot(300, self._initial_draw)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- שורת הכותרת (Header) ---
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame") # <--- הוספנו מזהה
        header_frame.setFixedHeight(40)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 0, 15, 0)

        lbl_title = QLabel("Cross Section Tool")
        lbl_title.setObjectName("HeaderTitle") # <--- הוספנו מזהה

        lbl_mode = QLabel(f"[{self.positioning_mode}]")
        lbl_mode.setObjectName("HeaderMode") # <--- הוספנו מזהה
        lbl_mode.setAlignment(Qt.AlignCenter)

        lbl_spacer = QLabel("")

        header_layout.addWidget(lbl_title, 1)
        header_layout.addWidget(lbl_mode, 1)
        header_layout.addWidget(lbl_spacer, 1)

        main_layout.addWidget(header_frame, 0)

        # --- אזור התוכן המפוצל ---
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        # הפיכת QWidget ל-QFrame כדי לתמוך בגבולות בצורה טובה יותר
        side = QFrame()
        side.setObjectName("SidePanel") # <--- הוספנו מזהה (אין פה ימין/שמאל, זה פשוט הפאנל של חלון החיתוך)
        side.setFixedWidth(260)
        sl = QVBoxLayout(side)

        sl.addWidget(QLabel(f"<b>Object:</b> {self.filename}"))
        sl.addSpacing(10)

        # ... (קוד הסליידר נשאר אותו דבר) ...

        # 1. נותנים מזהה ל-GroupBox
        gb = QGroupBox("Cut Position")
        gb.setObjectName("CutPositionGroup")

        vbox = QVBoxLayout()

        # 2. נותנים מזהה לתווית ה-Z ומוחקים את ה-setStyleSheet
        self.lbl_z = QLabel(f"Z: {self.current_z:.2f}")
        self.lbl_z.setAlignment(Qt.AlignCenter)
        self.lbl_z.setObjectName("LabelZValue")  # <-- המזהה החדש
        vbox.addWidget(self.lbl_z)

        # ... (כפתורי החיצים והסליידר נשארים כפי שהם, אין להם setStyleSheet)
        # זכור שנתנו לחיצים את המזהה "SmallArrowBtn" בשלב הקודם



        self.btn_z_up = QPushButton("▲")
        self.btn_z_up.setFixedSize(40, 30)
        self.btn_z_up.setAutoRepeat(True)
        self.btn_z_up.setAutoRepeatDelay(200)
        self.btn_z_up.setAutoRepeatInterval(30)
        self.btn_z_up.clicked.connect(self._step_z_up)

        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.setMinimumHeight(300)
        self.slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.btn_z_down = QPushButton("▼")
        self.btn_z_down.setFixedSize(40, 30)
        self.btn_z_down.setAutoRepeat(True)
        self.btn_z_down.setAutoRepeatDelay(200)
        self.btn_z_down.setAutoRepeatInterval(30)
        self.btn_z_down.clicked.connect(self._step_z_down)

        self.btn_z_up.setObjectName("SmallArrowBtn")
        self.btn_z_down.setObjectName("SmallArrowBtn")

        slider_vbox = QVBoxLayout()
        slider_vbox.addWidget(self.btn_z_up, 0, Qt.AlignHCenter)
        slider_vbox.addSpacing(10)
        slider_vbox.addWidget(self.slider, 1, Qt.AlignHCenter)
        slider_vbox.addSpacing(10)
        slider_vbox.addWidget(self.btn_z_down, 0, Qt.AlignHCenter)

        vbox.addLayout(slider_vbox)

        vbox.addSpacing(10)
        self.btn_auto_cuts = QPushButton("Auto Cuts ▼")


        auto_menu = QMenu(self.btn_auto_cuts)
        auto_menu.addAction("80% Height (Top Green Line)", lambda: self._apply_auto_cut(0.8))
        auto_menu.addAction("50% Height (Mid Green Line)", lambda: self._apply_auto_cut(0.5))
        auto_menu.addAction("20% Height (Bot Green Line)", lambda: self._apply_auto_cut(0.2))
        auto_menu.addSeparator()
        auto_menu.addAction("Max Width (Red Line)", self._apply_max_width_cut)
        auto_menu.addSeparator()
        auto_menu.addAction("Center of Mass (Z = 0.00)", self._apply_com_cut)
        self.btn_auto_cuts.setMenu(auto_menu)
        vbox.addWidget(self.btn_auto_cuts)

        gb.setLayout(vbox)
        sl.addWidget(gb)
        sl.addSpacing(20)

        btn_confirm = QPushButton("Confirm Cut")

        btn_confirm.clicked.connect(self.on_confirm_cut)
        sl.addWidget(btn_confirm)

        sl.addStretch()

        self.btn_help = QPushButton("Guide / Help")

        self.btn_help.clicked.connect(self.show_help_manual)
        sl.addWidget(self.btn_help)

        sl.addWidget(QLabel("1. Click on object to set height.\n2. Click 'Confirm Cut' to measure."))

        # מוסיפים את הפאנל השמאלי ללייאאוט התוכן (במקום ללייאאוט הראשי)
        content_layout.addWidget(side)

        self.live_views = LiveFiveViewsWidget(self.mesh, parent=self, locked=True)
        self.live_views.pointClicked.connect(self._on_3d_click)

        self.live_views.setStyleSheet("border: none; background-color: #f0f0f0;")
        if self.live_views.layout() is not None:
            self.live_views.layout().setSpacing(0)
            self.live_views.layout().setContentsMargins(0, 0, 0, 0)
        for internal_layout in self.live_views.findChildren(QLayout):
            internal_layout.setSpacing(0)
            internal_layout.setContentsMargins(0, 0, 0, 0)

        for lbl in self.live_views._coord_labels.values():
            lbl.hide()
            lbl.deleteLater()
        self.live_views._coord_labels.clear()

        # מוסיפים את המודלים של התלת-ממד ללייאאוט התוכן
        content_layout.addWidget(self.live_views, 1)
    def _initial_draw(self):
        self.draw_calipers()

        target_views = ['MC', 'ML', 'MR']
        for key in target_views:
            plotter = self.live_views._plotters.get(key)
            if plotter:
                actor = plotter.add_mesh(
                    self.cut_mesh,
                    color='black',
                    line_width=8,  # קו שחור עבה ובולט!
                    render_lines_as_tubes=True,
                    lighting=False,
                    pickable=False
                )
                mapper = actor.GetMapper()
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -66000)

        self.update_cut_line(force=True)

    def _on_slider_changed(self, val):
        ratio = val / 1000.0
        self.current_z = self.z_min + ratio * (self.z_max - self.z_min)

        self.lbl_z.setText(f"Z: {self.current_z:.2f}")
        self.update_cut_line()
        self._update_timer.start(100)

    def _step_z_up(self):
        current_val = self.slider.value()
        if current_val < self.slider.maximum():
            self.slider.setValue(current_val + 1)

    def _step_z_down(self):
        current_val = self.slider.value()
        if current_val > self.slider.minimum():
            self.slider.setValue(current_val - 1)

    def _apply_auto_cut(self, height_ratio):
        target_z = self.z_min + height_ratio * (self.z_max - self.z_min)
        self._set_z_value(target_z)

    def _apply_max_width_cut(self):
        """חיתוך אוטומטי בגובה של המיתר הרוחבי המקסימלי (הקו האדום)"""
        # מוודאים שנתוני המדידה קיימים
        if not getattr(self, "measurements", None):
            return

        # מושכים את הנתונים של פרופיל XZ (תצוגת MC הראשי)
        xz_profile = self.measurements.profile_xz
        if not xz_profile or not xz_profile.matlab_data:
            return

        matlab_data = xz_profile.matlab_data

        # pCordw מכיל את הקואורדינטות [X, Z] של תחילת המיתר המקסימלי
        if matlab_data.pCordw is not None:
            # האינדקס [1] הוא הציר האנכי של הפרופיל, שהוא בעצם ציר Z במרחב האמיתי
            target_z = float(matlab_data.pCordw[1])
            self._set_z_value(target_z)

    def _apply_com_cut(self):
        self._set_z_value(0.0)

    def _set_z_value(self, target_z):
        target_z = max(self.z_min, min(self.z_max, target_z))
        ratio = (target_z - self.z_min) / (self.z_max - self.z_min)
        slider_val = int(ratio * 1000)
        self.slider.setValue(slider_val)

    def _on_3d_click(self, view_key, x, y, z):
        self.current_z = max(self.z_min, min(self.z_max, z))
        self.lbl_z.setText(f"Z: {self.current_z:.2f}")

        if self.z_max > self.z_min:
            ratio = (self.current_z - self.z_min) / (self.z_max - self.z_min)
            blk = self.slider.blockSignals(True)
            self.slider.setValue(int(ratio * 1000))
            self.slider.blockSignals(blk)

        self.update_cut_line(force=True)

    def update_cut_line(self, force=False):
        if self._is_slicing and not force:
            return

        self._is_slicing = True

        try:
            slice_mesh = self.mesh.slice(normal=[0, 0, 1], origin=[0, 0, self.current_z])
            if slice_mesh.n_points > 0:
                self.cut_mesh.copy_from(slice_mesh)
            else:
                self.cut_mesh.copy_from(pv.PolyData())

            for key in ['MC', 'ML', 'MR']:
                plotter = self.live_views._plotters.get(key)
                if plotter: plotter.render()
        except Exception:
            pass

        self._is_slicing = False

    def draw_calipers(self):
        if not self.measurements: return
        m = self.measurements
        c = self.mesh.center

        if 'meas' not in self.actors_cache:
            self.actors_cache['meas'] = []

        def add_viz_line(views, p1, p2, color):
            for vk in views:
                plotter = self.live_views._plotters.get(vk)
                if plotter:
                    actor = _create_overlay_line(p1, p2, color, 3)
                    plotter.renderer.AddActor2D(actor)
                    self.actors_cache['meas'].append((vk, actor))

        D_xz = m.profile_xz.matlab_data
        if D_xz:
            y = c[1]
            if D_xz.pCord20p is not None: add_viz_line(['MC'], [D_xz.pCord20p[0], y, D_xz.pCord20p[1]],
                                                       [D_xz.pCord20p[0] + D_xz.cord20p, y, D_xz.pCord20p[1]], 'green')
            if D_xz.pCord80p is not None: add_viz_line(['MC'], [D_xz.pCord80p[0], y, D_xz.pCord80p[1]],
                                                       [D_xz.pCord80p[0] + D_xz.cord80p, y, D_xz.pCord80p[1]], 'green')
            if D_xz.pCordwhh is not None: add_viz_line(['MC'], [D_xz.pCordwhh[0], y, D_xz.pCordwhh[1]],
                                                       [D_xz.pCordwhh[0] + D_xz.cordwhh, y, D_xz.pCordwhh[1]], 'green')
            if D_xz.pCordw is not None:   add_viz_line(['MC'], [D_xz.pCordw[0], y, D_xz.pCordw[1]],
                                                       [D_xz.pCordw[0] + D_xz.cordw, y, D_xz.pCordw[1]], 'red')
            if D_xz.pCordh is not None:   add_viz_line(['MC'], [D_xz.pCordh[0], y, D_xz.pCordh[1]],
                                                       [D_xz.pCordh[0], y, D_xz.pCordh[1] - D_xz.cordh], 'red')

        D_yz = m.profile_yz.matlab_data
        if D_yz:
            x = c[0]
            if D_yz.pCord20p is not None: add_viz_line(['ML'], [x, D_yz.pCord20p[0], D_yz.pCord20p[1]],
                                                       [x, D_yz.pCord20p[0] + D_yz.cord20p, D_yz.pCord20p[1]], 'green')
            if D_yz.pCord80p is not None: add_viz_line(['ML'], [x, D_yz.pCord80p[0], D_yz.pCord80p[1]],
                                                       [x, D_yz.pCord80p[0] + D_yz.cord80p, D_yz.pCord80p[1]], 'green')
            if D_yz.pCordwhh is not None: add_viz_line(['ML'], [x, D_yz.pCordwhh[0], D_yz.pCordwhh[1]],
                                                       [x, D_yz.pCordwhh[0] + D_yz.cordwhh, D_yz.pCordwhh[1]], 'green')
            if D_yz.pCordw is not None:   add_viz_line(['ML'], [x, D_yz.pCordw[0], D_yz.pCordw[1]],
                                                       [x, D_yz.pCordw[0] + D_yz.cordw, D_yz.pCordw[1]], 'red')
            if D_yz.pCordh is not None:   add_viz_line(['ML'], [x, D_yz.pCordh[0], D_yz.pCordh[1]],
                                                       [x, D_yz.pCordh[0], D_yz.pCordh[1] - D_yz.cordh], 'red')

        for p in self.live_views._plotters.values():
            p.render()

    def on_confirm_cut(self):
        self.current_contours_data = compute_hatach_contours(self.mesh, self.current_z)
        self.result_window.set_contours(self.current_contours_data)
        self.measure_points = []
        self.saved_measurements = []

        self.redraw_2d_view()

        # שידור התמונה והקונטורים למסך הראשי
        pixmap = self.result_window.canvas.grab()
        self.cut_confirmed_signal.emit(pixmap, self.current_contours_data, self.result_window.is_looking_up, self.current_z)

        # הצגת החלון - show() משחרר את הנעילה (בניגוד ל-exec_())
        self.result_window.show()
        self.result_window.raise_()
        self.result_window.activateWindow()
    def redraw_2d_view(self, full_redraw=True):
        ax = self.result_window.ax

        # --- חלק 1: ציור כבד (מתבצע רק כשמשנים את החתך) ---
        if full_redraw:
            ax.clear()
            ax.axis('off')
            ax.set_aspect('equal', anchor='C')

            b = self.mesh.bounds
            cx = (b[0] + b[1]) / 2.0
            cy = (b[2] + b[3]) / 2.0
            dx = b[1] - b[0]
            dy = b[3] - b[2]

            margin = 1.25
            max_span = max(dx, dy) * margin

            x_min = cx - max_span / 2.0
            x_max = cx + max_span / 2.0
            y_min = cy - max_span / 2.0
            y_max = cy + max_span / 2.0

            # View: Looking DOWN (default) -> X רגיל, View: Looking UP -> X הפוך (מראה)
            if self.result_window.is_looking_up:
                ax.set_xlim(x_max, x_min)
            else:
                ax.set_xlim(x_min, x_max)

            # ציר ה-Y תמיד חייב להיות רגיל כדי להתאים למטלב (Y עולה למעלה)
            ax.set_ylim(y_min, y_max)

            if not self.current_contours_data:
                ax.text(0.5, 0.5, "No Intersection", ha='center', transform=ax.transAxes, color='red')
            else:
                total_perimeter = 0.0

                for cnt in self.current_contours_data:
                    x = np.append(cnt[:, 0], cnt[0, 0])
                    y = np.append(cnt[:, 1], cnt[0, 1])
                    ax.plot(x, y, color='black', linestyle='-', linewidth=4)
                    ax.fill(x, y, color='#808080', alpha=1.0)

                    closed_points = np.column_stack((x, y))
                    diffs = np.diff(closed_points, axis=0)
                    dists = np.linalg.norm(diffs, axis=1)
                    total_perimeter += np.sum(dists)

                ax.text(0.02, 0.98, f"Perimeter: {total_perimeter:.2f} mm",
                        transform=ax.transAxes, ha='left', va='top',
                        color='black', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        else:
            # במקום למחוק הכל, אנחנו מסירים רק את האובייקטים של המדידות מהציור הקודם
            for artist in getattr(self, '_meas_artists', []):
                try:
                    artist.remove()
                except Exception:
                    pass

        self._meas_artists = []

        # --- חלק 2: ציור קל ומהיר (המדידות עצמן) ---
        for meas in self.saved_measurements:
            pts = meas['points']
            if meas['type'] == 'dist':
                line, = ax.plot(pts[:, 0], pts[:, 1], 'go-', markersize=5)
                self._meas_artists.append(line)
                d = np.linalg.norm(pts[1] - pts[0])
                mid = (pts[0] + pts[1]) / 2
                txt = ax.text(mid[0], mid[1], f" {d:.2f} mm", color='green', fontsize=10, fontweight='bold')
                self._meas_artists.append(txt)
            elif meas['type'] == 'angle':
                line, = ax.plot(pts[:, 0], pts[:, 1], 'bo-', markersize=5)
                self._meas_artists.append(line)
                ba = pts[0] - pts[1]
                bc = pts[2] - pts[1]
                ang = np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1)))
                txt = ax.text(pts[1, 0], pts[1, 1], f" {ang:.1f}°", color='blue', fontsize=10, fontweight='bold')
                self._meas_artists.append(txt)

        # הוספנו ציור של הנקודות הפעילות באדום כדי שיהיה לך חיווי ויזואלי מיידי ללחיצות
        if self.measure_points:
            pts = np.array(self.measure_points)
            line, = ax.plot(pts[:, 0], pts[:, 1], 'r+-', markersize=8)
            self._meas_artists.append(line)

        # draw_idle מתעדכן בצורה חלקה יותר מאשר draw הרגיל
        self.result_window.canvas.draw_idle()

    def show_help_manual(self):
            help_text = """<b>Cross-Section Analysis - Quick Guide</b><br><br>

     <b>About This Tool:</b><br>
     This tool allows you to virtually slice the 3D artifact at any given height (Z-axis) to extract a precise 2D contour, calculate its perimeter, and perform manual measurements.<br><br>

     <b>How to Use:</b><br>
     1. <b>Set Cut Level:</b> Click anywhere on the 3D models to instantly snap the black cut-line to that height, or use the vertical slider to fine-tune the Z position.<br>
     2. <b>Generate Cut:</b> Click the "Confirm Cut" button to calculate the slice and open the 2D view.<br>
     3. <b>Measure:</b> In the 2D window, Right-click to place measurement points (2 points for distance, 3 for angle). Left-click to lock the measurement or clear points.<br><br>

     <b>Auto Cuts Menu:</b><br>
     * <b>Green Lines:</b> Aligns the cut with the 80%, 50%, or 20% height markers of the artifact.<br>
     * <b>Center of Mass:</b> Snaps the cut directly to the object's physical center of gravity (Z = 0.00)."""

            QMessageBox.information(self, "Cross-Section Manual", help_text)

    def _on_2d_click(self, event):
        if event.xdata is None or event.ydata is None: return

        if event.button == 3:
            if len(self.measure_points) == 3:
                self.saved_measurements.append({'type': 'angle', 'points': np.array(self.measure_points)})
                self.measure_points = [(event.xdata, event.ydata)]
            else:
                self.measure_points.append((event.xdata, event.ydata))

        elif event.button == 1:
            if len(self.measure_points) == 2:
                self.saved_measurements.append({'type': 'dist', 'points': np.array(self.measure_points)})
                self.measure_points = []
            elif len(self.measure_points) == 3:
                self.saved_measurements.append({'type': 'angle', 'points': np.array(self.measure_points)})
                self.measure_points = []
            elif len(self.measure_points) == 1:
                self.measure_points = []
            else:
                self.saved_measurements = []
                self.measure_points = []

        # שולחים full_redraw=False כדי לא לרנדר את המודל מחדש על כל קליק
        self.redraw_2d_view(full_redraw=False)

