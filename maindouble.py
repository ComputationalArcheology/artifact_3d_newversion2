# src/artifact_app/gui/main_window.py
from __future__ import annotations
from PySide6.QtWidgets import QApplication
from artifact_app.processing.process_object_measurements import compute_process_object_measurements
from artifact_app.processing.export_utils import export_batch_to_excel
from datetime import datetime
import pandas as pd
import os
from typing import Any

import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QSizePolicy, QLabel,
    QFileDialog, QMessageBox
)
from artifact_app.gui.cross_section_window import CrossSectionWindow
from artifact_app.gui.live_views_window import LiveViewsWindow
from artifact_app.gui.process_object_window import ProcessObjectWindow
from artifact_app.gui.manual_measurements_window import ManualMeasurementsWindow
from artifact_app.gui.widgets import ArtifactButton
from artifact_app.gui.crop_window import CropWindow
from artifact_app.gui.cross_section_window import CrossSectionWindow
from artifact_app.viewer.view_matlab_style import (
    MatlabFiveViewsCanvas,
    render_views_pixmaps_by_sizes,
)
from artifact_app.viewer.scale_bar import make_scale_pixmap
from artifact_app.viewer.main_view import compute_view_dirs_from_azel
from artifact_app.viewer.views_spec import get_views_spec

from artifact_app.processing.center_of_mass import compute_area_weighted_centroid
from artifact_app.processing.position import align_mesh


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artifact3_D")
        self.resize(1280, 860)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_settled)

        self._bbox_on: bool = False
        self._bbox_cache: dict[str, Any] | None = None
        self._com_on: bool = False
        self._com_view_keys: tuple[str, ...] = ("MC", "ML")
        self._pixmaps_before_com: dict[str, QPixmap] = {}

        self._last_mesh_raw: pv.PolyData | None = None
        self._last_mesh_aligned: pv.PolyData | None = None

        self._last_params = dict(
            background="white",
            color="silver",
            dist_scale=1.9,
            zoom_fact=1.00,
            multi_samples=8,
        )
        self._supersample = 4.0

        self._last_meta: dict[str, dict] = {}
        self._last_pixmaps: dict[str, QPixmap] = {}
        self._last_frame_mc: tuple[QPixmap, dict] | None = None

        self._process_window: ProcessObjectWindow | None = None
        self.live_views_win = None
        self.crop_win = None

        central = QWidget(self)
        self.setCentralWidget(central)
        central.setStyleSheet("background-color:#DCE1E8;")

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.side_left = QFrame(central)
        self.side_left.setFrameShape(QFrame.StyledPanel)
        self.side_left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.side_left.setFixedWidth(200)
        self.side_left.setStyleSheet("background-color: #8A9AAB; border-right: 1px solid #ccc;")

        self.side_left_layout = QVBoxLayout(self.side_left)
        self.side_left_layout.setContentsMargins(8, 8, 8, 8)
        self.side_left_layout.setSpacing(8)

        root.addWidget(self.side_left, 0)

        content = QWidget(central)
        content_v = QVBoxLayout(content)
        content_v.setContentsMargins(0, 0, 0, 0)
        content_v.setSpacing(0)
        root.addWidget(content, 1)

        self.label_filename = QLabel("", content)
        self.label_filename.setAlignment(Qt.AlignCenter)
        self.label_filename.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #444; background-color: #8596A8; padding: 4px;")
        content_v.addWidget(self.label_filename)

        self.views_box = QFrame(content)
        self.views_box.setStyleSheet("background:#DCE1E8; padding:0; margin:0; border:none;")
        self.views_box.setFrameShape(QFrame.NoFrame)
        self.views_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content_v.addWidget(self.views_box, 1)
        box_layout = QVBoxLayout(self.views_box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(0)

        self.five_views_canvas = MatlabFiveViewsCanvas(self.views_box)
        self.five_views_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        box_layout.addWidget(self.five_views_canvas, 1)

        self.labels = self.five_views_canvas.labels

        side = QFrame(central)
        side.setFrameShape(QFrame.StyledPanel)
        side.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        side.setFixedWidth(200)
        side.setStyleSheet("background-color: #8A9AAB; border-left: 1px solid #ccc;")
        side_v = QVBoxLayout(side)
        side_v.setContentsMargins(8, 8, 8, 8)
        side_v.setSpacing(8)
        root.addWidget(side, 0)

        self.btn_capture = ArtifactButton("Capture", side)
        self.btn_capture.setStyleSheet("background-color: #001F3F; color: white; font-weight: bold;")

        self.btn_refresh = ArtifactButton("Restore", side)

        self.btn_crop = ArtifactButton("Crop Object", side)
        self.btn_crop.setStyleSheet("background-color: #ffe0b2;")

        self.btn_mark_com_both = ArtifactButton("Mark center of mass", side)
        self.btn_dimensiones = ArtifactButton("dimensiones", side)
        self.btn_view_control = ArtifactButton("View Control", side)

        self.btn_build_bbox = ArtifactButton("Grid", side)

        self.btn_view_control.setStyleSheet("""
        QPushButton {
            background-color: #001F3F;
            color: white;
            border: 1px solid #888;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:checked {
            background-color: #2E5BFF;
            color: white;
            border: 1px solid #4466aa;
            font-weight: bold;
        }
        """)

        self.btn_hide_front = ArtifactButton("Front", side)
        self.btn_hide_front.clicked.connect(lambda: self.hide_view("MC"))

        self.btn_hide_right = ArtifactButton("Right", side)
        self.btn_hide_right.clicked.connect(lambda: self.hide_view("ML"))

        self.btn_hide_back = ArtifactButton("Back", side)
        self.btn_hide_back.clicked.connect(lambda: self.hide_view("MR"))

        self.btn_hide_top = ArtifactButton("Top", side)
        self.btn_hide_top.clicked.connect(lambda: self.hide_view("TL"))

        self.btn_hide_bottom = ArtifactButton("Bottom", side)
        self.btn_hide_bottom.clicked.connect(lambda: self.hide_view("BR"))

        self.btn_hide_title = ArtifactButton("Title", side)

        self.btn_hide_scale = ArtifactButton("Scale Bar", side)
        self.btn_hide_scale.clicked.connect(lambda: self.hide_view("SCALE"))

        self.btn_hide_title.clicked.connect(self._toggle_title_text)

        self.view_controls_box = QFrame(side)
        self.view_controls_box.setFrameShape(QFrame.NoFrame)
        view_controls_layout = QVBoxLayout(self.view_controls_box)
        view_controls_layout.setContentsMargins(0, 0, 0, 0)
        view_controls_layout.setSpacing(4)

        view_controls_layout.addWidget(self.btn_hide_front)
        view_controls_layout.addWidget(self.btn_hide_right)
        view_controls_layout.addWidget(self.btn_hide_back)
        view_controls_layout.addWidget(self.btn_hide_top)
        view_controls_layout.addWidget(self.btn_hide_bottom)
        view_controls_layout.addWidget(self.btn_build_bbox)
        view_controls_layout.addWidget(self.btn_hide_title)
        view_controls_layout.addWidget(self.btn_hide_scale)

        self.view_controls_box.setVisible(False)
        self.btn_view_control.setCheckable(True)
        self.btn_view_control.toggled.connect(self.view_controls_box.setVisible)
        self.btn_create_cut = ArtifactButton("Create cut", side)
        self.btn_create_cut.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_create_cut.clicked.connect(self.open_cross_section_window)
        side_v.addWidget(self.btn_capture)
        side_v.addWidget(self.btn_refresh)
        side_v.addWidget(self.btn_crop)
        side_v.addWidget(self.btn_mark_com_both)
        side_v.addWidget(self.btn_dimensiones)
        side_v.addWidget(self.btn_view_control)
        side_v.addWidget(self.view_controls_box)
        side_v.addWidget(self.btn_create_cut)
        side_v.addStretch(1)

        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_build_bbox.clicked.connect(self._on_build_bbox_clicked)
        self.btn_mark_com_both.clicked.connect(lambda: self.mark_center_of_mass_on_views(("MC", "ML")))
        self.btn_capture.clicked.connect(self.save_image_dialog)
        self.btn_dimensiones.clicked.connect(self.dimensiones_clicked)
        self.btn_crop.clicked.connect(self.open_crop_window)

        self._create_menubar()

    def _create_menubar(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("File")
        act_open = QAction("Open file…", self)
        act_open.triggered.connect(self.on_open)
        m_file.addAction(act_open)

        act_open_folder = QAction("Process Directory", self)
        act_open_folder.triggered.connect(self.on_open_folder)
        m_file.addAction(act_open_folder)
        m_file.addSeparator()
        m_file.addAction(QAction("Open folder", self))
        m_file.addAction(QAction("About file", self))

        m_position = menubar.addMenu("position")
        act_manual = QAction("manual positioning", self)
        act_manual.triggered.connect(self.open_live_views)
        m_position.addAction(act_manual)

        m_functions = menubar.addMenu("functions")
        m_functions.addAction(QAction("viewer", self))
        act_process = QAction("process object", self)
        act_process.triggered.connect(self.open_process_object_window)
        m_functions.addAction(act_process)
        act_manualmeasure = QAction("manual measurments", self)
        m_functions.addAction(act_manualmeasure)
        act_manualmeasure.triggered.connect(self.open_manual_measurements)

        m_extras = menubar.addMenu("extras")
        m_extras.addAction(QAction("Show/Hide Center of Mass", self))

        m_about = menubar.addMenu("About")
        m_about.addAction(QAction("Manual", self))
        m_about.addAction(QAction("Credits", self))

    def open_crop_window(self):
        mesh = getattr(self, "_last_mesh_aligned", None)
        if mesh is None:
            QMessageBox.warning(self, "No object", "אין אובייקט לחיתוך. טען קובץ קודם.")
            return

        self.crop_win = CropWindow(mesh, parent=None)
        self.crop_win.mesh_applied.connect(self.on_crop_applied)
        self.crop_win.show()

    def on_open_folder(self):
        import pathlib

        src_path = QFileDialog.getExistingDirectory(self, "Step 1: Select Source Directory (PLY files)")
        if not src_path:
            return

        base_dir = pathlib.Path(src_path)
        files = [f for f in base_dir.iterdir() if f.suffix.lower() == ".ply"]

        if not files:
            QMessageBox.warning(self, "Empty Directory", "No .ply files found in the selected directory.")
            return

        dest_path = QFileDialog.getExistingDirectory(self, "Step 2: Select Destination Directory")
        if not dest_path:
            return

        output_dir = pathlib.Path(dest_path)

        reply = QMessageBox.question(
            self,
            "Start Processing",
            f"Found {len(files)} files.\n\n"
            f"Source: {base_dir}\n"
            f"Destination: {output_dir}\n\n"
            f"Start processing?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.run_batch_processing(files, output_dir)

    def run_batch_processing(self, files, output_dir):
        """
        תהליך עיבוד אצווה (Batch Processing) אוטומטי מלא כולל חיווי זמן (ETA).
        """
        from artifact_app.processing.export_utils import export_batch_to_excel
        from PySide6.QtWidgets import QProgressDialog, QApplication
        from PySide6.QtCore import Qt
        import time

        dir_main = output_dir / "Screenshots_MainView"
        dir_graphs = output_dir / "Screenshots_Graphs"

        dir_main.mkdir(parents=True, exist_ok=True)
        dir_graphs.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("Preparing files...", "Cancel", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setWindowTitle("Batch Processing")
        progress.resize(400, 100)

        all_objects_data = []
        start_time = time.time()

        dummy_graph_window = ProcessObjectWindow(None, parent=self)
        dummy_graph_window.resize(1200, 800)
        dummy_graph_window.setAttribute(Qt.WA_DontShowOnScreen)
        dummy_graph_window.show()

        for i, file_path in enumerate(files):
            if progress.wasCanceled():
                break

            if i > 0:
                elapsed = time.time() - start_time
                avg_per_file = elapsed / i
                remaining_files = len(files) - i
                eta_seconds = int(avg_per_file * remaining_files)

                mins, secs = divmod(eta_seconds, 60)
                time_str = f"{mins} min {secs} sec"
            else:
                time_str = "Calculating..."

            progress.setValue(i)
            progress.setLabelText(f"Processing: {file_path.name}\n"
                                  f"File {i + 1} of {len(files)}\n"
                                  f"Estimated time: {time_str}")

            QApplication.processEvents()

            try:
                self.filename = file_path.name
                self.label_filename.setText(file_path.name)
                self.load_mesh_and_show(str(file_path))

                QApplication.processEvents()

                pix_main = self.views_box.grab()
                pix_main.save(str(dir_main / f"{file_path.stem}.png"))

                if self._last_mesh_aligned is not None:
                    dummy_graph_window.mesh = self._last_mesh_aligned
                    dummy_graph_window.filename = file_path.name
                    dummy_graph_window.findChild(QLabel, "").setText(file_path.name)

                    dummy_graph_window.update_measurements()

                    pix_graphs = dummy_graph_window.grab()
                    pix_graphs.save(str(dir_graphs / f"{file_path.stem}.png"))

                    m = dummy_graph_window._last_measurements
                    if m:
                        row = {
                            "Name": file_path.name,
                            "Volume": m.mesh_volume,
                            "Length (Height)": m.bind3box.dims['dz'],
                            "Width": m.bind3box.dims['dx'],
                            "Thickness": m.bind3box.dims['dy'],
                        }

                        if m.profile_xz.matlab_data:
                            d = m.profile_xz.matlab_data
                            row.update({
                                "XZ_Max_Horiz": m.profile_xz.max_chord,
                                "XZ_Max_Vert": d.cordh,
                                "XZ_Width_20%": d.cord20p,
                                "XZ_Width_50%": d.cordwhh,
                                "XZ_Width_80%": d.cord80p,
                            })

                        if m.profile_yz.matlab_data:
                            d = m.profile_yz.matlab_data
                            row.update({
                                "YZ_Max_Horiz": m.profile_yz.max_chord,
                                "YZ_Max_Vert": d.cordh,
                                "YZ_Width_20%": d.cord20p,
                                "YZ_Width_50%": d.cordwhh,
                                "YZ_Width_80%": d.cord80p,
                            })

                        all_objects_data.append(row)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        dummy_graph_window.close()
        progress.setValue(len(files))

        if all_objects_data:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            excel_path = output_dir / f"Batch_Summary_{timestamp}.xlsx"
            export_batch_to_excel(all_objects_data, str(excel_path))

            QMessageBox.information(
                self,
                "Processing Complete",
                f"Process finished successfully!\n\n"
                f"Images and report saved to:\n{output_dir}"
            )

        self._set_status("Batch processing finished.")

    def _generate_batch_screenshot(self, mesh, save_path):
            from PySide6.QtGui import QImage, QPainter, QColor
            from PySide6.QtCore import QRect
            from artifact_app.viewer.views_spec import get_views_spec

            total_w, total_h = 1200, 900

            layout_spec = {
                'MC': QRect(0, 200, 800, 500),
                'TL': QRect(0, 0, 800, 200),
                'BR': QRect(0, 700, 800, 200),
                'ML': QRect(800, 200, 400, 500),
                'MR': QRect(800, 0, 400, 200)
            }

            sizes_by_key = {k: (r.width(), r.height()) for k, r in layout_spec.items()}

            pixmaps = render_views_pixmaps_by_sizes(
                mesh,
                sizes_by_key=sizes_by_key,
                background="white",
                color="silver",
                dist_scale=1.9,
                zoom_fact=1.0,
                multi_samples=4,
                supersample=1.5,
                background_by_key={},
                meta_out={},
                views_spec=get_views_spec(),
                points_world=None
            )

            result_img = QImage(total_w, total_h, QImage.Format_RGB32)
            result_img.fill(QColor("white"))

            painter = QPainter(result_img)

            for key, rect in layout_spec.items():
                if key in pixmaps:
                    pm = pixmaps[key]
                    painter.drawPixmap(rect.x(), rect.y(), pm)
                    painter.setPen(QColor("#cccccc"))
                    painter.drawRect(rect)
                    painter.setPen(QColor("black"))
                    painter.drawText(rect.x() + 10, rect.y() + 20, key)

            painter.end()
            result_img.save(save_path)

    def _save_silent_screenshot(self, mesh, path):
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, color="silver")
        plotter.view_isometric()
        plotter.screenshot(path)
        plotter.close()

    def on_crop_applied(self, new_mesh):
        if new_mesh is None or new_mesh.n_points == 0:
            return

        print("Updating main view with cropped mesh...")

        self._last_mesh_aligned = new_mesh

        try:
            dx, dy, dz = bbox_deltas_from_bounds(new_mesh.bounds)
            self.five_views_canvas.set_bbox(dx, dy, dz)
        except Exception as e:
            print(f"Error updating bbox: {e}")

        self.refresh_static_views(new_mesh)

        self._bbox_on = False
        self._bbox_cache = None
        self._com_on = False
        self._on_build_bbox_clicked()

        self._set_status("Object cropped successfully.")

    def _set_status(self, text: str) -> None:
        print(text)

    def _current_dpr(self) -> float:
        try:
            wh = self.window().windowHandle() if self.window() else None
            scr = wh.screen() if wh else None
            return float(scr.devicePixelRatio()) if scr else 1.0
        except Exception:
            return 1.0

    def on_open(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "בחר קובץ תלת-ממדי",
                "",
                "3D Files (*.ply *.stl *.obj *.wrl *.vtk *.vtp)",
            )
            if not path:
                return
            self.filename = os.path.basename(path)
            self.label_filename.setText(self.filename)
            self.load_mesh_and_show(path)
        except Exception as e:
            QMessageBox.critical(self, "Open error", str(e))

    def open_manual_measurements(self) -> None:
        mesh = getattr(self, "_last_mesh_aligned", None)
        if mesh is None:
            QMessageBox.warning(self, "No object", "אין mesh מיושר. טען אובייקט קודם.")
            return

        path_name = ""
        obj_name = getattr(self, "filename", "object")
        w = ManualMeasurementsWindow(
            mesh,
            path_name=path_name,
            object_name=obj_name,
            postype="inertia",
            parent=self,
        )
        w.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        w.setAttribute(Qt.WA_DeleteOnClose, True)
        w.show()

    def load_mesh_and_show(self, path: str) -> None:
        try:
            mesh = pv.read(path).clean(inplace=False)
            self._last_mesh_raw = mesh.copy()

            mesh_aligned, _Ttr = align_mesh(mesh, method="matlab")

            try:
                mesh_aligned = mesh_aligned.compute_normals(
                    point_normals=True,
                    cell_normals=False,
                    splitting=False,
                    auto_orient_normals=True,
                    feature_angle=120.0,
                )
            except Exception:
                pass

            self._last_mesh_aligned = mesh_aligned
            self._original_aligned_mesh = mesh_aligned.copy(deep=True)

            dx, dy, dz = bbox_deltas_from_bounds(mesh_aligned.bounds)
            self.five_views_canvas.set_bbox(dx, dy, dz)

            self.refresh_static_views(mesh_aligned)

            self._bbox_on = False
            self._bbox_cache = None
            self._on_build_bbox_clicked()

        except Exception as e:
            print(f"[error] load_mesh_and_show: {e}")
            self._set_status(str(e))

    def dimensiones_clicked(self):
        mesh = self._last_mesh_aligned
        if mesh is None:
            self._set_status("no object to show")
            return

        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        text = f"Object dimensions:\nX = {dx:.2f}\nY = {dy:.2f}\nZ = {dz:.2f}"
        QMessageBox.information(self, "Object size", text)

    def hide_view(self, view_key: str) -> None:
        lbl = self.labels.get(view_key)
        if lbl is None:
            return

        hidden = getattr(lbl, "_logically_hidden", False)

        if not hidden:
            lbl.clear()
            lbl._logically_hidden = True
            return

        pix = self._last_pixmaps.get(view_key)
        if pix is not None:
            lbl.setPixmap(pix)
        lbl._logically_hidden = False

    def refresh_static_views(self, mesh_aligned: pv.PolyData) -> None:
        if mesh_aligned is None:
            return

        was_bbox_on = bool(getattr(self, "_bbox_on", False))
        was_com_on = bool(getattr(self, "_com_on", False))
        com_keys = tuple(getattr(self, "_com_view_keys", ("MC", "ML")))

        self._last_mesh_aligned = mesh_aligned
        self._last_meta = {}

        sizes = self.five_views_canvas.get_view_sizes()

        ss = float(getattr(self, "_supersample", 1.0))
        ss = 1.0 if ss < 1.0 else ss

        tmp = dict(getattr(self, "_last_params", {}))
        color = tmp.get("color", "silver")
        dist_scale = float(tmp.get("dist_scale", 1.9))
        zoom_fact = float(tmp.get("zoom_fact", 1.0))
        multi_samples = int(tmp.get("multi_samples", 0))

        try:
            com, _vol = compute_area_weighted_centroid(mesh_aligned)
        except Exception as e:
            self._set_status(f"Center-of-Mass error: {e}")
            com = None

        bg_other = "#DCE1E8"
        background_by_key = {
            "ML": bg_other,
            "MR": bg_other,
            "TL": bg_other,
            "BR": bg_other,
        }

        pixmaps = render_views_pixmaps_by_sizes(
            mesh_aligned,
            sizes_by_key=sizes,
            background="white",
            color=color,
            dist_scale=dist_scale,
            zoom_fact=zoom_fact,
            multi_samples=multi_samples,
            supersample=ss,
            background_by_key=background_by_key,
            meta_out=self._last_meta,
            views_spec=get_views_spec(),
            points_world=({"com": com} if com is not None else None),
        )

        self._last_pixmaps = {}
        for key, pix in pixmaps.items():
            lbl = self.labels.get(key)
            self._last_pixmaps[key] = pix

            if lbl is None or getattr(lbl, "_logically_hidden", False):
                continue
            lbl.setPixmap(pix)

        mc_meta = self._last_meta.get("MC")
        if isinstance(mc_meta, dict) and "MC" in self._last_pixmaps:
            self._last_frame_mc = (self._last_pixmaps["MC"], mc_meta)

        if was_com_on:
            self._apply_com_overlay_only(com_keys, com_override=com)
            self._com_on = True
        else:
            self._com_on = False

        if was_bbox_on:
            self._bbox_on = True
            self._apply_bbox_overlay_only()

        try:
            self._update_scale_slot()
        except Exception as e:
            self._set_status(f"[scale] skipped: {e}")

    def _apply_com_overlay_only(self, view_keys=("MC", "ML"), com_override=None) -> None:
        if self._last_mesh_aligned is None:
            return

        try:
            if com_override is not None:
                com = com_override
            else:
                com, _vol = compute_area_weighted_centroid(self._last_mesh_aligned)
        except Exception:
            return

        for vk in view_keys:
            pix = self._last_pixmaps.get(vk)
            meta = self._last_meta.get(vk)
            lbl = self.labels.get(vk)

            if pix is None or not isinstance(meta, dict):
                continue
            if lbl is not None and getattr(lbl, "_logically_hidden", False):
                continue

            try:
                rp = meta.get("proj_points", {})
                if isinstance(rp, dict) and "com" in rp:
                    u, v = rp["com"]
                else:
                    u, v = self._world_to_mr_pixels(com, meta, flip_v=True)

                pix_drawn = self._draw_crosshair_on_pixmap(pix, int(u), int(v), radius=5)
                self._last_pixmaps[vk] = pix_drawn
                if lbl is not None:
                    lbl.setPixmap(pix_drawn)
            except Exception:
                pass

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_mesh_aligned is None:
            return
        self._resize_timer.start(150)

    def _on_resize_settled(self) -> None:
        if self._last_mesh_aligned is None:
            return
        self.refresh_static_views(self._last_mesh_aligned)

    def _on_refresh_clicked(self) -> None:
        """
        שחזור למצב ראשוני. משחזר את המודל לגרסתו המקורית, מאפס העדפות,
        ומבטל את כל מצבי ההסתרה של התצוגות.
        """
        if getattr(self, "_original_aligned_mesh", None) is None:
            if getattr(self, "_last_mesh_raw", None) is not None:
                self._set_status("Performing full realign (slow)...")
                mesh = self._last_mesh_raw.copy()
                mesh_aligned, _ = align_mesh(mesh, method="matlab")
                self._original_aligned_mesh = mesh_aligned
            else:
                return

        self._set_status("Restoring view...")

        self._last_mesh_aligned = self._original_aligned_mesh.copy(deep=True)

        self._last_params = dict(
            background="white",
            color="silver",
            dist_scale=1.9,
            zoom_fact=1.00,
            multi_samples=8,
        )

        if hasattr(self, "labels") and self.labels:
            for lbl in self.labels.values():
                if hasattr(lbl, "_logically_hidden"):
                    lbl._logically_hidden = False

        self._bbox_on = False
        self._com_on = False
        self._bbox_cache = None
        self._pixmaps_before_com = {}

        dx, dy, dz = bbox_deltas_from_bounds(self._last_mesh_aligned.bounds)
        self.five_views_canvas.set_bbox(dx, dy, dz)

        self.refresh_static_views(self._last_mesh_aligned)

        self._on_build_bbox_clicked()

        self._set_status("Restored: Geometry & Views reset.")

    def _calculate_bbox_via_image_scan(self, pixmap: QPixmap) -> tuple[int, int, int, int] | None:
        img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        width = img.width()
        height = img.height()

        ptr = img.constBits()
        bpl = img.bytesPerLine()
        try:
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bpl))
            arr = arr[:, :width * 3].reshape((height, width, 3))
        except Exception:
            return None

        bg_color = arr[0, 0]
        mask = np.any(arr != bg_color, axis=2)

        if not np.any(mask):
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return int(x_min), int(x_max) + 1, int(y_min), int(y_max) + 1

    def _toggle_title_text(self):
        if self.label_filename.text().strip():
            self.label_filename.setText(" ")
        else:
            name = getattr(self, "filename", "")
            self.label_filename.setText(name)

    def _on_build_bbox_clicked(self) -> None:
        if getattr(self, "_bbox_on", False):
            frame = getattr(self, "_last_frame_mc", None)
            if frame and isinstance(frame, tuple) and len(frame) == 2:
                base_pix, mc_meta = frame
                lbl_mc = self.labels.get("MC")
                if lbl_mc is not None and not getattr(lbl_mc, "_logically_hidden", False):
                    lbl_mc.setPixmap(base_pix)

                self._last_pixmaps["MC"] = base_pix
                if isinstance(mc_meta, dict):
                    mc_meta.pop("bbox_px", None)
                    mc_meta.pop("bbox_span_mm", None)

            self._bbox_on = False
            self._bbox_cache = None
            self._set_status("BBox off")
            return

        if self._last_mesh_aligned is None:
            self._set_status("אין mesh מיושר — טען/יישר קודם.")
            return

        frame = getattr(self, "_last_frame_mc", None)
        if not frame or not isinstance(frame, tuple) or len(frame) != 2:
            self._set_status("אין בסיס MC לציור — לחץ restore ואז נסה שוב.")
            return

        base_pix, mc_meta = frame
        if not isinstance(mc_meta, dict):
            return

        try:
            bbox_px = self._calculate_bbox_via_image_scan(base_pix)
            if bbox_px is None:
                w, h = base_pix.width(), base_pix.height()
                bbox_px = (0, w, 0, h)

            dx_mm, dz_mm = self._span_mm_in_view_axes(self._last_mesh_aligned, mc_meta)

            mc_meta["bbox_px"] = bbox_px
            mc_meta["bbox_span_mm"] = (dx_mm, dz_mm)

            pix_ov = self._draw_bbox_rulers_simple(
                base_pix,
                bbox_px=bbox_px,
                dx_mm=dx_mm,
                dz_mm=dz_mm,
            )

            lbl_mc = self.labels.get("MC")
            if lbl_mc is not None and not getattr(lbl_mc, "_logically_hidden", False):
                lbl_mc.setPixmap(pix_ov)

            self._last_pixmaps["MC"] = pix_ov
            self._bbox_on = True

            self._bbox_cache = {"mode": "px", "bbox_px": bbox_px, "bbox_span_mm": (dx_mm, dz_mm)}

        except Exception as e:
            self._set_status(f"BBox overlay error: {e}")

    def _apply_bbox_overlay_only(self) -> None:
        if not getattr(self, "_bbox_on", False):
            return
        if self._last_mesh_aligned is None:
            return

        frame = getattr(self, "_last_frame_mc", None)
        if not frame or not isinstance(frame, tuple) or len(frame) != 2:
            return

        base_pix, mc_meta = frame

        try:
            bbox_px = self._calculate_bbox_via_image_scan(base_pix)
            if bbox_px is None:
                return

            dx_mm, dz_mm = self._span_mm_in_view_axes(self._last_mesh_aligned, mc_meta)

            mc_meta["bbox_px"] = bbox_px
            mc_meta["bbox_span_mm"] = (dx_mm, dz_mm)

            pix_ov = self._draw_bbox_rulers_simple(
                base_pix,
                bbox_px=bbox_px,
                dx_mm=dx_mm,
                dz_mm=dz_mm,
            )

            self._last_pixmaps["MC"] = pix_ov
            lbl_mc = self.labels.get("MC")
            if lbl_mc is not None and not getattr(lbl_mc, "_logically_hidden", False):
                lbl_mc.setPixmap(pix_ov)

        except Exception as e:
            self._set_status(f"BBox reapply error: {e}")

    def _basis_from_azel_roll(self, az: float, el: float, roll: float):
        right, up, fwd = compute_view_dirs_from_azel(az, el)

        rr = np.deg2rad(float(roll))
        if rr != 0.0:
            c, s = np.cos(rr), np.sin(rr)
            r2 = c * right + s * up
            u2 = -s * right + c * up
            right = r2 / (np.linalg.norm(r2) or 1.0)
            up = u2 / (np.linalg.norm(u2) or 1.0)

        return right, up, fwd

    def _world_to_mr_pixels(self, pt3, meta: dict, flip_v: bool = True):
        ps = float(meta["parallel_scale"])
        w, h = map(int, meta["window_size"])
        cx, cy, cz = map(float, meta.get("center", (0.0, 0.0, 0.0)))

        if "right" in meta and "up" in meta:
            right = np.asarray(meta["right"], float)
            up = np.asarray(meta["up"], float)
            right /= (np.linalg.norm(right) + 1e-12)
            up /= (np.linalg.norm(up) + 1e-12)
        else:
            az = float(meta.get("az", 0.0))
            el = float(meta.get("el", 0.0))
            roll = float(meta.get("roll", 0.0))
            right, up, _ = self._basis_from_azel_roll(az, el, roll)
            right = np.asarray(right, float)
            up = np.asarray(up, float)
            right /= (np.linalg.norm(right) + 1e-12)
            up /= (np.linalg.norm(up) + 1e-12)

        d = np.asarray(pt3, float) - np.array([cx, cy, cz], float)
        xw = float(np.dot(d, right))
        yw = float(np.dot(d, up))

        aspect = (w / h) if h else 1.0
        x_half_world = ps * aspect

        u = int((xw / (2 * x_half_world) + 0.5) * w)
        v = int((yw / (2 * ps) + 0.5) * h)
        if flip_v:
            v = h - v
        return u, v

    def _draw_crosshair_on_pixmap(self, pix: QPixmap, u: int, v: int, radius: int = 4) -> QPixmap:
        pix2 = pix.copy()
        p = QPainter(pix2)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(Qt.red))
        p.drawEllipse(QPoint(u, v), radius, radius)
        p.end()
        return pix2

    def _nice_step(self, span: float, target_ticks: int = 6) -> float:
        if span <= 0:
            return 1.0
        raw = span / max(target_ticks, 1)
        pow10 = 10.0 ** np.floor(np.log10(raw))
        for m in (1.0, 2.0, 5.0, 10.0):
            if raw <= m * pow10:
                return m * pow10
        return 10.0 * pow10

    def _span_mm_in_view_axes(self, mesh: pv.PolyData, meta: dict) -> tuple[float, float]:
        if mesh is None or mesh.n_points <= 0:
            return 0.0, 0.0
        center = np.asarray(meta.get("center", mesh.center), float)
        if "right" in meta and "up" in meta:
            right = np.asarray(meta["right"], float)
            up = np.asarray(meta["up"], float)
        else:
            az = float(meta.get("az", 0.0))
            el = float(meta.get("el", 0.0))
            roll = float(meta.get("roll", 0.0))
            right, up, _ = self._basis_from_azel_roll(az, el, roll)
        right = np.asarray(right, float)
        up = np.asarray(up, float)
        right /= (np.linalg.norm(right) + 1e-12)
        up /= (np.linalg.norm(up) + 1e-12)
        d = mesh.points.astype(float) - center[None, :]
        u = d @ right
        v = d @ up
        dx = float(u.max() - u.min())
        dz = float(v.max() - v.min())
        return max(0.0, dx), max(0.0, dz)

    def _draw_bbox_rulers_simple(
            self,
            pix: QPixmap,
            *,
            bbox_px: tuple[int, int, int, int],
            dx_mm: float,
            dz_mm: float,
    ) -> QPixmap:
        if pix is None or pix.isNull():
            return pix

        obj_left, obj_right, obj_top, obj_bottom = map(int, bbox_px)
        W, H = pix.width(), pix.height()

        obj_left = max(0, min(obj_left, W - 1))
        obj_right = max(0, min(obj_right, W - 1))
        obj_top = max(0, min(obj_top, H - 1))
        obj_bottom = max(0, min(obj_bottom, H - 1))

        w_px = float(obj_right - obj_left)
        h_px = float(obj_bottom - obj_top)

        if w_px < 3 or h_px < 3:
            return pix
        if dx_mm <= 0 or dz_mm <= 0:
            return pix

        LINE_W = 1.0
        TICK_LEN = 4
        GAP = 2
        FONT_PT = 6
        BG_COLOR = QColor("#DCE1E8")

        step_x = self._nice_step(dx_mm, target_ticks=7)
        step_z = self._nice_step(dz_mm, target_ticks=9)
        ticks_x = np.arange(0.0, dx_mm + 1e-9, step_x)
        ticks_z = np.arange(0.0, dz_mm + 1e-9, step_z)

        show_cm_x = (dx_mm >= 10.0)
        show_cm_z = (dz_mm >= 10.0)

        pix2 = QPixmap(pix)
        painter = QPainter(pix2)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        font = QFont()
        font.setPointSize(FONT_PT)
        painter.setFont(font)
        fm = painter.fontMetrics()

        box_left = obj_left
        box_right = obj_right
        box_top = obj_top
        box_bottom = obj_bottom

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(BG_COLOR))

        if box_top > 0:
            painter.drawRect(0, 0, W, box_top)

        if box_bottom < H:
            painter.drawRect(0, box_bottom, W, H - box_bottom)

        if box_left > 0:
            painter.drawRect(0, box_top, box_left, box_bottom - box_top)

        if box_right < W:
            painter.drawRect(box_right, box_top, W - box_right, box_bottom - box_top)

        pen_line = QPen(Qt.black, LINE_W)
        pen_line.setCosmetic(True)
        painter.setPen(pen_line)
        painter.setBrush(Qt.NoBrush)

        painter.drawLine(box_left, box_top, box_left, box_bottom)
        painter.drawLine(box_left, box_bottom, box_right, box_bottom)
        painter.drawLine(box_right, box_top, box_right, box_bottom)
        painter.drawLine(box_left, box_top, box_right, box_top)

        box_height = box_bottom - box_top
        px_per_mm_z = box_height / max(1e-9, dz_mm)

        for val_mm in ticks_z:
            v = int(round(box_bottom - val_mm * px_per_mm_z))
            if v < box_top - 2 or v > box_bottom + 2:
                continue

            painter.drawLine(box_left, v, box_left + TICK_LEN, v)

            label = (f"{val_mm / 10:g}" if show_cm_z else f"{val_mm:g}")
            text_w = fm.horizontalAdvance(label)
            text_x = box_left - GAP - text_w
            text_x = max(0, text_x)
            text_y = v + fm.ascent() // 2
            text_y = max(fm.ascent(), min(text_y, H - 1))
            painter.drawText(text_x, text_y, label)

        box_width = box_right - box_left
        px_per_mm_x = box_width / max(1e-9, dx_mm)

        for val_mm in ticks_x:
            u = int(round(box_left + val_mm * px_per_mm_x))
            if u < box_left - 2 or u > box_right + 2:
                continue

            painter.drawLine(u, box_bottom, u, box_bottom - TICK_LEN)

            label = (f"{val_mm / 10:g}" if show_cm_x else f"{val_mm:g}")
            text_w = fm.horizontalAdvance(label)
            text_x = u - text_w // 2
            text_x = max(0, min(text_x, W - text_w))
            text_y = box_bottom + GAP + fm.ascent()
            text_y = min(text_y, H - 1)
            painter.drawText(text_x, text_y, label)

        painter.end()
        return pix2

    def _on_view_applied(self, mesh, rot_matrix):
        self._last_mesh_aligned = mesh
        self._manual_rot_matrix = rot_matrix
        self.refresh_static_views(mesh)

    def mark_center_of_mass_on_views(self, view_keys=("MC", "ML")):
        self._com_view_keys = tuple(view_keys)
        if getattr(self, "_com_on", False):
            for vk in view_keys:
                base_pix = getattr(self, "_pixmaps_before_com", {}).get(vk)
                if base_pix is None:
                    continue
                self._last_pixmaps[vk] = base_pix
                lbl = self.labels.get(vk)
                if lbl is not None and not getattr(lbl, "_logically_hidden", False):
                    lbl.setPixmap(base_pix)
            self._com_on = False
            self._pixmaps_before_com = {}
            self._set_status("Center of Mass OFF")
            return

        if self._last_mesh_aligned is None:
            self._set_status("אין mesh מיושר — טען/יישר קודם.")
            return

        try:
            com, _vol = compute_area_weighted_centroid(self._last_mesh_aligned)
        except Exception as e:
            self._set_status(f"Center-of-Mass error: {e}")
            return

        self._pixmaps_before_com = {}
        shown_any = False

        for vk in view_keys:
            pix = self._last_pixmaps.get(vk)
            meta = self._last_meta.get(vk)
            lbl = self.labels.get(vk)

            if pix is None or not isinstance(meta, dict):
                continue
            if lbl is not None and getattr(lbl, "_logically_hidden", False):
                continue

            self._pixmaps_before_com[vk] = pix

            try:
                rp = meta.get("proj_points", {})
                if isinstance(rp, dict) and "com" in rp:
                    u, v = rp["com"]
                else:
                    u, v = self._world_to_mr_pixels(com, meta, flip_v=True)

                pix_drawn = self._draw_crosshair_on_pixmap(pix, int(u), int(v), radius=5)
                self._last_pixmaps[vk] = pix_drawn
                if lbl is not None:
                    lbl.setPixmap(pix_drawn)

                shown_any = True
            except Exception:
                pass

        if shown_any:
            self._com_on = True
        else:
            self._com_on = False
            self._pixmaps_before_com = {}

    def open_live_views(self):
        """פתיחת חלון Manual Positioning בטוחה"""
        mesh = getattr(self, "_last_mesh_aligned", None)
        if mesh is None:
            QMessageBox.warning(self, "אין אובייקט", "אין mesh מיושר להצגה.")
            return

        if self.live_views_win is not None:
            try:
                self.live_views_win.close()
            except:
                pass
            self.live_views_win = None

        self.live_views_win = LiveViewsWindow(mesh, parent=None)
        self.live_views_win.viewApplied.connect(self._on_live_view_applied)

        self.live_views_win.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.live_views_win.setAttribute(Qt.WA_DeleteOnClose, True)
        self.live_views_win.show()

    def _on_live_view_applied(self, params: dict) -> None:
        """
        קבלת המש המסובב מחלון Manual Positioning ותמיכה גם בגישה החדשה
        (קבלת קודקודים מסובבים) וגם בישנה (חישוב מ-fwd/up).
        """
        print("[main_window] _on_live_view_applied called")

        if self._last_mesh_aligned is None:
            self._set_status("אין mesh מיושר.")
            return

        try:
            rotated_vertices = params.get("rotated_vertices")

            if rotated_vertices is not None:
                print("[main_window] Using rotated vertices from live_views")

                faces = params.get("original_faces", self._last_mesh_aligned.faces)
                mesh_rot = pv.PolyData(rotated_vertices, faces)

                mesh_rot = mesh_rot.clean()

                try:
                    mesh_rot = mesh_rot.compute_normals(
                        point_normals=True,
                        cell_normals=True,
                        auto_orient_normals=True,
                    )
                except Exception as e:
                    print(f"[main_window] compute_normals warning: {e}")
                    try:
                        mesh_rot = mesh_rot.compute_normals(auto_orient_normals=True)
                    except:
                        pass

                self._last_mesh_aligned = mesh_rot

                rot_matrix = params.get("rot_matrix")
                if rot_matrix is not None:
                    self._manual_rot_matrix = np.array(rot_matrix)

                dx, dy, dz = bbox_deltas_from_bounds(mesh_rot.bounds)
                self.five_views_canvas.set_bbox(dx, dy, dz)

                self.refresh_static_views(mesh_rot)
                self._set_status("Manual positioning applied.")

                print("[main_window] Mesh updated successfully")

                if self.live_views_win is not None and self.live_views_win.isVisible():
                    self.live_views_win.update_mesh_and_reset(mesh_rot)

                return

            print("[main_window] Fallback: computing rotation from fwd/up")

            fwd = params.get("fwd")
            up = params.get("up")

            if fwd is None or up is None:
                az_live = float(params.get("az", 0.0))
                el_live = float(params.get("el", 0.0))
                roll_live = float(params.get("roll", 0.0))
                F_front = self._frame_from_angles(az_live, el_live, roll_live)
            else:
                F_front = self._frame_from_fwd_up(np.asarray(fwd, float), np.asarray(up, float))

            mc_meta = self._last_meta.get("MC", {})
            az0 = float(mc_meta.get("az", 0.0))
            el0 = float(mc_meta.get("el", 0.0))
            roll0 = float(mc_meta.get("roll", 0.0))
            F_mc_base = self._frame_from_angles(az0, el0, roll0)

            R_delta = F_front @ F_mc_base.T
            R_obj = R_delta.T

            pts = self._last_mesh_aligned.points.astype(float)
            center = np.array([
                (self._last_mesh_aligned.bounds[1] + self._last_mesh_aligned.bounds[0]) / 2.0,
                (self._last_mesh_aligned.bounds[3] + self._last_mesh_aligned.bounds[2]) / 2.0,
                (self._last_mesh_aligned.bounds[5] + self._last_mesh_aligned.bounds[4]) / 2.0
            ])

            pts_centered = pts - center
            pts_rot = (R_obj @ pts_centered.T).T
            pts_final = pts_rot + center

            mesh_rot = pv.PolyData(pts_final, self._last_mesh_aligned.faces)
            mesh_rot = mesh_rot.clean()

            try:
                mesh_rot = mesh_rot.compute_normals(auto_orient_normals=True)
            except:
                pass

            self._last_mesh_aligned = mesh_rot

            dx, dy, dz = bbox_deltas_from_bounds(mesh_rot.bounds)
            self.five_views_canvas.set_bbox(dx, dy, dz)

            self.refresh_static_views(mesh_rot)
            self._set_status("Manual positioning applied.")

            if self.live_views_win is not None and self.live_views_win.isVisible():
                self.live_views_win.update_mesh_and_reset(mesh_rot)

        except Exception as e:
            print(f"[main_window] Error in _on_live_view_applied: {e}")
            import traceback
            traceback.print_exc()
            self._set_status(f"Error: {e}")

    @staticmethod
    def _frame_from_angles(az: float, el: float, roll: float) -> np.ndarray:
        right, up, fwd = compute_view_dirs_from_azel(float(az), float(el))
        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        up = up / (np.linalg.norm(up) or 1.0)

        if roll:
            axis = fwd / (np.linalg.norm(fwd) or 1.0)
            theta = np.deg2rad(float(roll))
            v = up
            v_par = np.dot(v, axis) * axis
            v_perp = v - v_par
            w = np.cross(axis, v_perp)
            up = v_par + v_perp * np.cos(theta) + w * np.sin(theta)
            up = up / (np.linalg.norm(up) or 1.0)

        right = np.cross(up, fwd)
        right = right / (np.linalg.norm(right) or 1.0)
        up = np.cross(fwd, right)

        return np.column_stack([right, up, fwd])

    @staticmethod
    def _frame_from_fwd_up(fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = np.asarray(fwd, float)
        u = np.asarray(up, float)
        f /= (np.linalg.norm(f) or 1.0)
        u /= (np.linalg.norm(u) or 1.0)

        r = np.cross(u, f)
        r /= (np.linalg.norm(r) or 1.0)
        u = np.cross(f, r)

        return np.column_stack([r, u, f])

    def _update_scale_slot(self) -> None:
        """
        שומרת בזיכרון את תמונת הסרגל כדי שפונקציית hide_view תוכל לשחזר אותה
        """
        lbl = self.labels.get("SCALE")
        if lbl is None:
            return
        if getattr(lbl, "_logically_hidden", False):
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
        if "window_size" not in ref_meta:
            return

        mesh = getattr(self, "_last_mesh_aligned", None)
        if mesh is None or mesh.n_points <= 0:
            return

        center = np.asarray(ref_meta.get("center", mesh.center), float)
        right = np.asarray(ref_meta["right"], float)
        right /= (np.linalg.norm(right) + 1e-12)
        u = (mesh.points.astype(float) - center[None, :]) @ right
        obj_w_mm = float(u.max() - u.min())

        pm = make_scale_pixmap(
            w, h,
            ref_meta,
            target_frac=0.80,
            dpr=self._current_dpr(),
            s0_mm=obj_w_mm,
            debug=True,
            bg="#DCE1E8",
        )
        self._last_pixmaps["SCALE"] = pm
        lbl.setPixmap(pm)

    def save_image_dialog(self) -> None:
        pm = self.views_box.grab()
        suggest = getattr(self, "filename", "capture_grid")
        default_path = os.path.join(os.path.expanduser("~"), f"{suggest}.png")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "שמור תמונה בשם...",
            default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;Bitmap (*.bmp)",
        )
        if not path:
            return
        if os.path.splitext(path)[1] == "":
            path += ".png"
        if not pm.save(path):
            QMessageBox.critical(self, "שגיאת שמירה", "שמירה נכשלה.")
            return
        QMessageBox.information(self, "נשמר!", f"התמונה נשמרה בהצלחה:\n{path}")

    def open_cross_section_window(self) -> None:
        """פותח את חלון החיתוך התלת-ממדי ישירות מהמסך הראשי, כולל הקווים המנחים"""
        mesh = getattr(self, "_last_mesh_aligned", None)
        if mesh is None:
            QMessageBox.warning(self, "No object", "אין אובייקט מיושר. טען קובץ קודם.")
            return

        filename = getattr(self, "filename", "")

        # --- הוקוס פוקוס! חישוב המדידות (הקווים) מאחורי הקלעים ---
        self._set_status("Calculating measurements for cross section...")
        mc_meta = self._last_meta.get("MC", {})

        try:
            # חישוב המדידות כדי שהקווים הירוקים והאדומים יופיעו בחיתוך!
            # הגדלים (500,500) הם פיקטיביים ולא משפיעים על התלת-ממד.
            measurements = compute_process_object_measurements(
                mesh,
                mc_meta=mc_meta,
                view_size_xz=(500, 500),
                view_size_yz=(500, 500),
                height_percent=20.0
            )
        except Exception as e:
            print(f"Error computing measurements: {e}")
            measurements = None
        # -----------------------------------------------------------

        # יוצרים את החלון ומעבירים אליו את המדידות שחישבנו עכשיו
        self.cross_section_win = CrossSectionWindow(
            mesh=mesh,
            measurements=measurements,  # <--- הנה הקסם שמצייר את הקווים!
            filename=filename,
            parent=None  # Parent=None מנתק אותו מהחלון הראשי כדי שיוכל לרוץ עצמאית
        )

        self.cross_section_win.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.cross_section_win.show()
        self._set_status("Cross section window opened.")

    def open_process_object_window(self) -> None:
        mesh = getattr(self, "_last_mesh_aligned", None)
        if mesh is None:
            QMessageBox.warning(self, "No object", "אין mesh מיושר. טען אובייקט קודם.")
            return

        mc_meta = self._last_meta.get("MC", {})
        ml_meta = self._last_meta.get("ML", {})
        filename = getattr(self, "filename", "")

        w = ProcessObjectWindow(
            mesh_aligned=mesh,
            mc_meta=mc_meta,
            ml_meta=ml_meta,
            parent=self,
            filename=filename,
        )
        w.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        w.setWindowModality(Qt.NonModal)
        w.show()
        self._process_window = w




def bbox_deltas_from_bounds(bounds) -> tuple[float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    dz = float(zmax - zmin)
    return dx, dy, dz