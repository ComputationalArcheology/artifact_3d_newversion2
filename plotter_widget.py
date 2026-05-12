from PySide6.QtWidgets import QWidget, QVBoxLayout
from pyvistaqt import QtInteractor
import pyvista as pv

class PlotterWidget(QWidget):
    """עטיפה ל-QtInteractor להצגת מודלים תלת-ממדיים עם תאורה ושיידינג."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        # הגדרות תצוגה בסיסיות
        self.plotter.set_background("white")
        #self.plotter.show_axes()
        self.plotter.camera.parallel_projection = False  # תמיד עומק

        # תאורה + EDL (אם נתמך)
        self.setup_lights()
        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        self._actor = None

    def setup_lights(self):
        """שלוש נקודות אור בסיסיות ליצירת עומק טוב."""
        self.plotter.remove_all_lights()
        key = pv.Light(position=(1.5, 1.5, 2.5), focal_point=(0, 0, 0), intensity=1.0)
        fill = pv.Light(position=(-2.0, -1.0, 1.5), focal_point=(0, 0, 0), intensity=0.4)
        rim = pv.Light(position=(0.0, -3.0, -1.0), focal_point=(0, 0, 0), intensity=0.6)
        self.plotter.add_light(key)
        self.plotter.add_light(fill)
        self.plotter.add_light(rim)

    def show_mesh(self, mesh, color="lightgray"):
        """מציג mesh בודד עם שיידינג נעים."""
        self.plotter.clear()
        self.setup_lights()
        self._actor = self.plotter.add_mesh(
            mesh,
            color=color,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.8,
            specular=0.35,
            specular_power=20,
        )
        self.plotter.camera.parallel_projection = False
        self.plotter.reset_camera()
        self.plotter.render()
        return self._actor  # ← תוספת יחידה, לא משנה את הנראות
