# src/artifact_app/viewer/main_view.py
from __future__ import annotations
import numpy as np


def compute_view_dirs_from_azel(az: float, el: float):
    """
    מחזיר וקטורים (right, up, fwd) במערכת הצירים העולמית עבור אזימוט (az) והעלאה (el).
    0° => +Y ; 90° => +X
    fwd מצביע מהמצלמה אל האובייקט (Camera -> Focal Point).
    """
    azr = np.deg2rad(float(az))
    elr = np.deg2rad(float(el))

    fwd_xy = np.array([np.sin(azr), np.cos(azr), 0.0], dtype=float)
    R_el = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elr), -np.sin(elr)],
            [0.0, np.sin(elr), np.cos(elr)],
        ],
        dtype=float,
    )
    fwd = R_el @ fwd_xy
    fwd /= (np.linalg.norm(fwd) or 1.0)

    up = np.array([0.0, 0.0, 1.0], float)
    if abs(float(np.dot(fwd, up))) > 0.99:
        up = np.array([0.0, 1.0, 0.0], float)

    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) or 1.0)

    return right, up, fwd


# ============================================================
# בחירת חזית אוטומטית (Auto Front)
# ============================================================
def _choose_front_azel_from_bounds(bounds) -> tuple[float, float]:
    """
    מחשב זווית "Front" אוטומטית על ידי בחירת הציר הדק ביותר (min span).
    מונע מצב שהמודל מוצג כ"גפרור" דק מהצד.
    מחזיר: (az, el).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)

    # במקרה של שוויון (למשל X≈Z), שבירת שוויון עקבית לפי הסדר: X -> Z -> Y
    order = [0, 2, 1]
    min_span = float(np.min(spans))
    candidates = [ax for ax in order if abs(float(spans[ax]) - min_span) <= 1e-9]
    axis = int(candidates[0]) if candidates else int(np.argmin(spans))

    if axis == 0:
        return 90.0, 0.0
    if axis == 1:
        return 0.0, 0.0

    return 0.0, 90.0


def set_view_azel(
        plotter,
        az: float | None,
        el: float | None,
        center: np.ndarray,
        radius: float,
        *,
        ortho: bool = True,
        margin: float = 1.0,
        bounds=None,
) -> None:
    """
    מציב את המצלמה לפי אזימוט והעלאה.
    אם הועברו ערכי None, ניתן לשלב את זה בעתיד עם _choose_front_azel_from_bounds.
    """
    if az is None:
        az = 0.0
    if el is None:
        el = 0.0

    right, up, fwd = compute_view_dirs_from_azel(az, el)

    cam = plotter.camera
    cam.SetFocalPoint(*center)
    cam.SetPosition(*(center - fwd * (radius * 2.2)))
    cam.SetViewUp(*up)

    if ortho:
        plotter.enable_parallel_projection()
        cam.SetParallelScale(float(radius) * float(margin))
    else:
        plotter.disable_parallel_projection()


def _debug_print_cam(plotter, tag="cam"):
    cam = plotter.camera
    pos = np.asarray(cam.GetPosition(), float)
    focal = np.asarray(cam.GetFocalPoint(), float)
    up = np.asarray(cam.GetViewUp(), float)
    fwd = focal - pos
    fwd /= (np.linalg.norm(fwd) or 1.0)
    print(f"[{tag}] fwd≈{np.round(fwd, 3)}, up≈{np.round(up, 3)}, dot(up,+Z)={up[2]:.3f}")