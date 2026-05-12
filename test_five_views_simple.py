# -*- coding: utf-8 -*-
# test_five_views_simple.py
from __future__ import annotations
import sys
from typing import Dict, Tuple

import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout

# נייבא את המימוש המדויק של view(az,el) + התאמת פריימינג
from artifact_app.viewer.views_spec import set_view_azel

# הגדרות המבטים בסגנון Rotate_for_GUI (אזימוט, אלטיטודה)
VIEWS = {
    "TL": ( 90.0,  90.0),  # Top
    "ML": (-90.0,   0.0),  # Left
    "MC": (  0.0,   0.0),  # Front
    "MR": ( 90.0,   0.0),  # Right
    "BR": ( 90.0, -90.0),  # Bottom
}




class FiveViews(QWidget):
    """
    חלון 3×3 “חמישה מבטים” כמו במטלב:
    TL → (0,0), ML → (1,0), MC → (1,1), MR → (1,2), BR → (2,2)

    שים לב:
    - אנו משתמשים ב־set_view_azel מ־views_spec, שמקבל center/radius/bounds
      ומחשב ParallelScale/Framing באופן יציב ואורתוגרפי.
    """
    def __init__(self, mesh: pv.PolyData):
        super().__init__()
        self.setWindowTitle("Five MATLAB-like Views (lean)")

        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # חישובי פריימינג פעם אחת:
        bounds = mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
        center = mesh.center
        # רדיוס: חצי המקס' מכל הטווחים
        radius = 0.5 * max(bounds[1] - bounds[0],
                           bounds[3] - bounds[2],
                           bounds[5] - bounds[4]) or 1.0

        # מיקומים (שדות לא בשימוש אפשר להשאיר ריקים כדי ליצור “צלב” של 5 חלונות)
        slots = {
            "TL": (0, 0),
            "ML": (1, 0),
            "MC": (1, 1),
            "MR": (1, 2),
            "BR": (2, 2),
        }

        # יצירת הפלוטרים והצבת מבטים
        self.plots: Dict[str, QtInteractor] = {}
        for key, (row, col) in slots.items():
            p = QtInteractor(self)
            grid.addWidget(p, row, col)
            p.set_background("white")

            # רנדר בסיסי (פשוט לבדיקה)
            try:
                triang = mesh.triangulate(inplace=False)
            except Exception:
                triang = mesh
            p.add_mesh(
                triang,
                color="silver",
                smooth_shading=False,
                lighting=True,
                ambient=0.20, diffuse=0.80, specular=0.0,
                show_edges=True, edge_color="#2b2f36", line_width=1.0,
                reset_camera=False, render=False,
            )

            # קביעת המבט (אורתוגראפי) + פריימינג עקבי
            az, el = VIEWS[key]
            set_view_azel(
                p,
                az=az,
                el=el,
                center=center,
                radius=radius,
                ortho=True,
                bounds=bounds,
                margin=1.06,
            )

            # שיפורים קטנים לנראות (לא חובה)
            try:
                p.add_mesh(triang.outline(), color="#1c2129", line_width=0.9,
                           reset_camera=False, render=False)
            except Exception:
                pass

            try:
                p.enable_eye_dome_lighting()
            except Exception:
                pass

            p.render()
            self.plots[key] = p


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)

    # שימוש:
    #   python test_five_views_simple.py                 → כדור ברירת־מחדל
    #   python test_five_views_simple.py path/to/model   → טען קובץ שלך
    if len(sys.argv) > 1:
        path = sys.argv[1]
        mesh = pv.read(path).clean(inplace=False)
    else:
        # דוגמה בלבד. בפועל תספק mesh מיושר מ-align_mesh.
        mesh = pv.Sphere(theta_resolution=64, phi_resolution=64)

    w = FiveViews(mesh)
    w.resize(1200, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
