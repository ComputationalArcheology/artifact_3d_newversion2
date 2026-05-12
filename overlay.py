# src/artifact_app/viewer/overlay.py
from __future__ import annotations
import numpy as np
from PySide6.QtGui import QPixmap, QPainter, QPen, QImage, QColor
from PySide6.QtCore import Qt, QPointF
import pyvista as pv


def _view_basis_from_azel_roll(az: float, el: float, roll: float):
    """
    מחשב את בסיס המבט (right, up, fwd) לפי זוויות אזימוט, העלאה (elevation) וגלגול (roll).
    """
    azr = np.deg2rad(az); elr = np.deg2rad(el)
    fwd = np.array([np.cos(elr)*np.cos(azr), np.cos(elr)*np.sin(azr), np.sin(elr)], float)
    fwd /= (np.linalg.norm(fwd) or 1.0)
    up0 = np.array([0, 0, 1.0], float)
    if abs(float(np.dot(fwd, up0))) > 0.99:
        up0 = np.array([0, 1.0, 0], float)
    right = np.cross(up0, fwd); right /= (np.linalg.norm(right) or 1.0)
    up    = np.cross(fwd, right); up   /= (np.linalg.norm(up) or 1.0)
    if roll:
        th = np.deg2rad(roll); c, s = np.cos(th), np.sin(th)
        R2 = np.array([[c, -s], [s, c]], float)
        RU = np.stack([right, up], axis=1) @ R2
        right, up = RU[:, 0], RU[:, 1]
    return right, up, fwd

def _world_to_screen(X: np.ndarray, meta: dict, *, flip_v: bool = True) -> tuple[float, float]:
    """
    מבצע הקרנה אורתוגרפית של קואורדינטת תלת-ממד לקואורדינטת מסך (פיקסלים).
    """
    center = np.asarray(meta["center"], float)
    W, H = int(meta["window_size"][0]), int(meta["window_size"][1])
    ps = float(meta["parallel_scale"])

    if "right" in meta and meta["right"] is not None and "up" in meta and meta["up"] is not None:
        right = np.asarray(meta["right"], float)
        up    = np.asarray(meta["up"], float)
    else:
        az = float(meta.get("az", 90.0))
        el = float(meta.get("el", 0.0))
        roll = float(meta.get("roll", 0.0))
        right, up, _ = _view_basis_from_azel_roll(az, el, roll)

    d = np.asarray(X, float) - center
    u = float(np.dot(d, right))
    v = float(np.dot(d, up))
    x = (u / ps) * (W / 2.0) + (W / 2.0)
    y = (v / ps) * (H / 2.0) + (H / 2.0)
    if flip_v:
        y = H - y
    return x, y

def _poly3d_to_qpoints(poly_pts_3d: np.ndarray, meta: dict, *, flip_v=True):
    pts = [QPointF(*_world_to_screen(p, meta, flip_v=flip_v)) for p in poly_pts_3d]
    return pts

def paint_bbox_on_mr_pixmap(
    base_pix: QPixmap,
    r: np.ndarray | None,
    p: dict | None,
    meta: dict | None,
    *,
    flip_v: bool = True,
    color: Qt.GlobalColor | Qt.GlobalColor = Qt.red,
    width: int = 2,
) -> QPixmap:
    """
    מצייר תיבת BBox תלת-ממדית (הקרנה של 8 קודקודים) על גבי העותק של התמונה.
    """
    if base_pix is None or meta is None:
        return base_pix

    needed = ("center", "parallel_scale", "window_size")
    if any(k not in meta for k in needed):
        return base_pix

    pix = base_pix.copy()
    painter = QPainter(pix)
    pen = QPen(color)
    pen.setWidth(width)
    painter.setPen(pen)

    try:
        if r is not None and isinstance(r, np.ndarray) and r.ndim == 2 and r.shape[1] == 3:
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7),
            ]
            pts2 = [_world_to_screen(r[i], meta, flip_v=flip_v) for i in range(r.shape[0])]
            for i, j in edges:
                x1, y1 = pts2[i]; x2, y2 = pts2[j]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        if p and "center" in p:
            cx, cy = _world_to_screen(np.asarray(p["center"], float), meta, flip_v=flip_v)
            painter.drawEllipse(QPointF(cx, cy), 4.0, 4.0)

    except Exception:
        painter.end()
        return base_pix
    finally:
        painter.end()

    return pix

def paint_screen_bbox_from_mesh(
    base_pix: QPixmap,
    mesh: pv.PolyData,
    meta: dict,
    *,
    flip_v: bool = True,
    color: Qt.GlobalColor = Qt.red,
    width: int = 2,
    margin_px: int = 0,
) -> QPixmap:
    """
    מצייר מסגרת 2D (תיחום פשוט) סביב ההיטל של המודל על המסך (מחקה את המסגרת של MATLAB).
    """
    if base_pix is None or meta is None or mesh is None:
        return base_pix

    pts = mesh.points
    if pts is None or len(pts) == 0:
        return base_pix

    step = max(1, len(pts) // 5000)
    xs, ys = [], []
    for P in pts[::step]:
        x, y = _world_to_screen(P, meta, flip_v=flip_v)
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return base_pix

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xmin += margin_px
    xmax -= margin_px
    ymin += margin_px
    ymax -= margin_px

    W = base_pix.width()
    H = base_pix.height()
    xmin = max(0, min(W - 1, int(round(xmin))))
    xmax = max(0, min(W - 1, int(round(xmax))))
    ymin = max(0, min(H - 1, int(round(ymin))))
    ymax = max(0, min(H - 1, int(round(ymax))))

    if xmax <= xmin or ymax <= ymin:
        return base_pix

    pix = base_pix.copy()
    painter = QPainter(pix)
    pen = QPen(color)
    pen.setWidth(width)
    painter.setPen(pen)
    painter.drawRect(xmin, ymin, xmax - xmin, ymax - ymin)
    painter.end()
    return pix


def paint_bbox_from_pixmap_content(
    base_pix: QPixmap,
    *,
    color: Qt.GlobalColor = Qt.red,
    width: int = 2,
    bg: Qt.GlobalColor = Qt.white,
    sample_step: int = 2,
) -> QPixmap:
    """
    מצייר מלבן 2D על ידי סריקת הפיקסלים בתמונה (מזהה כל מה שאינו צבע הרקע).
    """

    if base_pix is None or base_pix.isNull():
        return base_pix

    img: QImage = base_pix.toImage().convertToFormat(QImage.Format_RGBA8888)
    W, H = img.width(), img.height()
    if W <= 0 or H <= 0:
        return base_pix

    bg_color = QColor(bg)
    bg_r, bg_g, bg_b, _ = bg_color.getRgb()

    xs: list[int] = []
    ys: list[int] = []

    step = max(1, int(sample_step))

    for y in range(0, H, step):
        for x in range(0, W, step):
            c = img.pixelColor(x, y)
            r, g, b, _ = c.getRgb()
            if abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b) > 10:
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return base_pix

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xmin = max(0, min(W - 1, xmin))
    xmax = max(0, min(W - 1, xmax))
    ymin = max(0, min(H - 1, ymin))
    ymax = max(0, min(H - 1, ymax))

    if xmax <= xmin or ymax <= ymin:
        return base_pix

    pix = base_pix.copy()
    painter = QPainter(pix)
    pen = QPen(color)
    pen.setWidth(width)
    painter.setPen(pen)
    painter.drawRect(xmin, ymin, xmax - xmin, ymax - ymin)
    painter.end()
    return pix