# src/artifact_app/gui/mesh_viewer.py
import pyvista as pv
import vtk  # נדרש עבור אירועי העכבר המתקדמים
import numpy as np
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt


class InteractiveMeshViewer(QWidget):
    def __init__(self, mesh, title="3D Coordinate Probe", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.status_bar = QFrame()
        self.status_bar.setStyleSheet("background-color: #f0f0f0; border-top: 1px solid #ccc;")
        self.status_bar.setFixedHeight(40)

        status_layout = QVBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)
        status_layout.setAlignment(Qt.AlignVCenter)

        # גופן מונוספייס מונע מהמספרים "לקפוץ" כשהם מתעדכנים בתצוגה
        self.lbl_coords = QLabel("Move mouse over object to probe coordinates...")
        self.lbl_coords.setStyleSheet("font-family: Consolas; font-size: 14px; color: #555;")
        status_layout.addWidget(self.lbl_coords)

        layout.addWidget(self.status_bar)

        self.mesh = mesh

        # CellPicker עדיף על PointPicker לריחוף כי הוא מדויק יותר על פני משטחים
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)

        # רדיוס הסמן נקבע באופן פרופורציונלי לגודל האובייקט (כ-1% מהאלכסון)
        if self.mesh and self.mesh.n_points > 0:
            self.cursor_radius = self.mesh.length * 0.01
        else:
            self.cursor_radius = 0.1

        self.cursor_actor = None

        self._init_scene()

    def _init_scene(self):
        self.plotter.set_background("white")

        if self.mesh:
            self.plotter.add_mesh(
                self.mesh,
                color="#E0E0E0",
                show_edges=False,
                smooth_shading=True,
                specular=0.5,
                opacity=1.0
            )

            # אופטימיזציה: במקום ליצור סמן מחדש בכל תזוזה (איטי), יוצרים פעם אחת ומסתירים
            cursor_mesh = pv.Sphere(radius=self.cursor_radius)
            self.cursor_actor = self.plotter.add_mesh(
                cursor_mesh,
                color='red',
                name='cursor_marker',
                opacity=0.8
            )
            self.cursor_actor.SetVisibility(False)

            self.plotter.show_axes()
            self.plotter.show_grid(color='#D0D0D0', font_size=10)

            # הוק לאירוע תזוזת עכבר - מחליף את enable_point_picking שלא עובד מספיק טוב לריחוף רציף
            self.plotter.interactor.AddObserver(vtk.vtkCommand.MouseMoveEvent, self._on_mouse_move)

        self.plotter.reset_camera()

    def _on_mouse_move(self, obj, event):
        x, y = self.plotter.interactor.GetEventPosition()

        self.picker.Pick(x, y, 0, self.plotter.renderer)

        cell_id = self.picker.GetCellId()

        if cell_id != -1:
            picked_pos = self.picker.GetPickPosition()

            text = f"X: {picked_pos[0]:.3f}   Y: {picked_pos[1]:.3f}   Z: {picked_pos[2]:.3f}"
            self.lbl_coords.setText(f"📍 {text}")
            self.lbl_coords.setStyleSheet("font-family: Consolas; font-size: 14px; color: #000; font-weight: bold;")

            self.cursor_actor.SetPosition(picked_pos)
            self.cursor_actor.SetVisibility(True)

        else:
            self.lbl_coords.setText("...")
            self.lbl_coords.setStyleSheet("font-family: Consolas; font-size: 14px; color: #999;")

            if self.cursor_actor:
                self.cursor_actor.SetVisibility(False)

        self.plotter.render()