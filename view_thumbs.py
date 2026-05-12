import pyvista as pv
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

def _np_to_qpixmap(img: np.ndarray, target: int = 192) -> QPixmap:
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("Unexpected image shape (expected HxWx3 or HxWx4)")
    h, w, ch = img.shape
    if ch == 3:
        img_rgb = img[..., ::-1].copy()             # BGR->RGB
        qimg = QImage(img_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
    else:
        img_rgba = img[..., [2,1,0,3]].copy()       # BGRA->RGBA
        qimg = QImage(img_rgba.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def _setup_thumb_lights(p: pv.Plotter):
    """תאורה כמו בפלוטר הראשי כדי לקבל אותו מראה."""
    p.remove_all_lights()
    key  = pv.Light(position=(1.5, 1.5, 2.5), focal_point=(0,0,0), intensity=1.0)
    fill = pv.Light(position=(-2.0, -1.0, 1.5), focal_point=(0,0,0), intensity=0.45)
    rim  = pv.Light(position=(0.0, -3.0, -1.0), focal_point=(0,0,0), intensity=0.6)
    p.add_light(key); p.add_light(fill); p.add_light(rim)
    try:
        p.enable_eye_dome_lighting()  # מוסיף עומק וקונטרסט
    except Exception:
        pass
    try:
        p.enable_anti_aliasing()
    except Exception:
        pass

def make_view_pixmap(mesh: pv.PolyData, view: str, *, size=(640, 640), ortho: bool = True) -> QPixmap:
    """
    מרנדר תמונת מבט סטטית (thumbnail) עם אותה תאורה/שיידינג כמו בחלון הראשי,
    ברזולוציה גבוהה ואז מקטין – כדי לקבל חדות וקונטרסט טובים.
    """
    p = pv.Plotter(off_screen=True, window_size=size)
    try:
        p.set_background("white")
        _setup_thumb_lights(p)
        p.add_mesh(
            mesh,
            color="#bfbfbf",            # מעט כהה מ-"lightgray" כדי לא להיראות דהוי
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.8,
            specular=0.35,
            specular_power=30,
        )

        v = (view or "").lower()
        if v == "front":   p.view_xz()
        elif v == "side":  p.view_yz()
        elif v == "top":   p.view_xy()
        else:              p.view_isometric()

        p.camera.parallel_projection = bool(ortho)
        img = p.screenshot(return_img=True)  # רינדור גדול → יקטן ל־192px
    finally:
        p.close()

    return _np_to_qpixmap(img, target=192)
