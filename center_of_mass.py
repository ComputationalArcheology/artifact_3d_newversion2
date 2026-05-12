# -*- coding: utf-8 -*-
# src/artifact_app/processing/center_of_mass.py
from __future__ import annotations

import numpy as np
import pyvista as pv


def _triangulate_compat(mesh: pv.PolyData) -> pv.PolyData:
    """
    טריאנגולציה תואמת-גרסאות על משטח נקי.
    מדלגת על הפעולה לחלוטין אם המודל כבר מורכב ממשולשים,
    כדי לחסוך זמן עיבוד משמעותי במודלים כבדים.
    """
    # הבדיקה החדשה: עוקף את פעולת ה-triangulate אם אין בה צורך
    if mesh.is_all_triangles:
        return mesh

    surf = mesh.extract_surface()
    if surf.n_points == 0:
        return surf

    try:
        return surf.triangulate(inplace=False)
    except TypeError:
        pass

    try:
        tri = surf.triangulate()
        if isinstance(tri, pv.PolyData):
            return tri
    except TypeError:
        pass

    m2 = surf.copy(deep=True)
    try:
        m2.triangulate(inplace=True)
    except TypeError:
        pass
    return m2
def _surface_centroid_fallback(mesh: pv.PolyData) -> np.ndarray:
    """
    צנטרואיד משטחי פשוט (ממוצע נקודות) - משמש כגיבוי למקרי קצה שבהם חישוב הנפח נכשל.
    """
    pts = np.asarray(mesh.points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    return pts.mean(axis=0)


def compute_area_weighted_centroid(mesh: pv.PolyData) -> tuple[np.ndarray, float]:
    """
    חישוב מרכז מסה נפחי (Volumetric Centroid).

    Returns:
        tuple[np.ndarray, float]: (מרכז_מסה, נפח_כולל)
    """
    if mesh is None or mesh.n_points == 0:
        raise ValueError("mesh ריק או לא תקין")

    tri = _triangulate_compat(mesh)
    if tri.n_points == 0:
        return np.zeros(3, dtype=np.float64), 0.0

    tri = tri.clean()
    pts = np.asarray(tri.points, dtype=np.float64)

    if pts.size == 0:
        return np.zeros(3, dtype=np.float64), 0.0

    faces = np.asarray(tri.faces, dtype=np.int64)
    if faces.size == 0:
        return _surface_centroid_fallback(tri), 0.0

    # המערך במבנה VTK כולל גם את מספר הקודקודים לפני כל פאה (למשל: [3, i0, i1, i2])
    if faces.size % 4 != 0:
        return _surface_centroid_fallback(tri), 0.0

    faces_reshape = faces.reshape(-1, 4)
    mask_tri = faces_reshape[:, 0] == 3
    if not np.any(mask_tri):
        return _surface_centroid_fallback(tri), 0.0

    tri_faces = faces_reshape[mask_tri, 1:]

    tris = pts[tri_faces]
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]

    # שימוש בממוצע גיאומטרי כנקודת ייחוס לשיפור יציבות נומרית בחישובים מול נקודות רחוקות
    p0 = pts.mean(axis=0)
    a0 = a - p0
    b0 = b - p0
    c0 = c - p0

    v_signed = np.einsum("ij,ij->i", a0, np.cross(b0, c0)) / 6.0

    mask_valid = np.isfinite(v_signed) & (np.abs(v_signed) > 0.0)
    if not np.any(mask_valid):
        return _surface_centroid_fallback(tri), 0.0

    v_signed = v_signed[mask_valid]
    tetra_centers = (p0[None, :] + a[mask_valid] + b[mask_valid] + c[mask_valid]) / 4.0

    V_total = v_signed.sum()

    if (not np.isfinite(V_total)) or (np.abs(V_total) < 1e-12):
        return _surface_centroid_fallback(tri), 0.0

    com = (v_signed[:, None] * tetra_centers).sum(axis=0) / V_total

    return com.astype(np.float64, copy=False), float(V_total)