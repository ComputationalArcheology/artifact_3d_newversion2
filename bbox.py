# artifact_app/processing/bbox.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pyvista as pv


# ---------------------------------------
# מבני נתונים
# ---------------------------------------

@dataclass
class CaliperSet:
    """
    מדידות מעטפת בסגנון MATLAB (לפי בסיס המבט):
      DD0 (XZ): psCalw = [x1, z@x1, x2, z@x2]
      DD1 (YZ): psCalw = [y1, z@y1, y2, z@y2] ; psCalh = [y@z1, z1, y@z2, z2]
    """
    psCalw: np.ndarray
    psCalh: np.ndarray


# ---------------------------------------
# פונקציות עזר לבניית בסיס המבט (View Basis)
# ---------------------------------------

def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v

_WORLD_X = np.array([1.0, 0.0, 0.0], dtype=float)
_WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=float)

def _canonicalize_basis(right: np.ndarray, up: np.ndarray, fwd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    מבטיח שהבסיס יהיה אורתונורמלי, ימני ובעל אוריינטציה יציבה מול צירי העולם.
    """
    right = _safe_unit(right)
    up    = _safe_unit(up)
    fwd   = _safe_unit(fwd)

    if float(np.dot(np.cross(right, up), fwd)) < 0.0:
        fwd = -fwd

    if float(np.dot(up, _WORLD_Z)) < 0.0:
        up = -up
        fwd = -fwd

    if float(np.dot(right, _WORLD_X)) < 0.0:
        right = -right
        fwd = -fwd

    right = _safe_unit(right - up * float(np.dot(right, up)))
    fwd   = _safe_unit(np.cross(right, up))
    up    = _safe_unit(np.cross(fwd, right))
    return right, up, fwd


def _basis_from_meta(mesh: pv.PolyData, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    מחלץ (right, up, fwd, center) מתוך נתוני המבט.
    מתעדף וקטורים קיימים במטא-דאטה ומשלים חישובים לפי אזימוט/העלאה במידת הצורך.
    """
    center = None
    if "center" in meta:
        try:
            c = np.asarray(meta["center"], dtype=float).reshape(3,)
            if np.isfinite(c).all():
                center = c
        except Exception:
            center = None

    if center is None:
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        center = np.array([(xmax + xmin) * 0.5, (ymax + ymin) * 0.5, (zmax + zmin) * 0.5], dtype=float)

    if "right" in meta and "up" in meta:
        right = _safe_unit(np.asarray(meta["right"], float))
        up0 = np.asarray(meta["up"], float)
        up = _safe_unit(up0 - right * float(np.dot(up0, right)))

        if "fwd" in meta:
            fwd0 = np.asarray(meta["fwd"], float)
            fwd = fwd0 - right * float(np.dot(fwd0, right)) - up * float(np.dot(fwd0, up))
            fwd = _safe_unit(fwd)
        else:
            fwd = _safe_unit(np.cross(right, up))

        if float(np.dot(np.cross(right, up), fwd)) < 0.0:
            fwd = -fwd

        right = _safe_unit(np.cross(up, fwd))
        up = _safe_unit(np.cross(fwd, right))

        right, up, fwd = _canonicalize_basis(right, up, fwd)
        return right, up, fwd, center

    from artifact_app.viewer.main_view import compute_view_dirs_from_azel

    az = float(meta.get("az", 0.0))
    el = float(meta.get("el", 0.0))
    roll = float(meta.get("roll", 0.0))

    right, up, fwd = compute_view_dirs_from_azel(az, el)
    right = _safe_unit(np.asarray(right, float))
    up = _safe_unit(np.asarray(up, float))
    fwd = _safe_unit(np.asarray(fwd, float))

    if roll:
        rr = np.deg2rad(roll)
        c, s = np.cos(rr), np.sin(rr)
        r2 = c * right + s * up
        u2 = -s * right + c * up
        right = _safe_unit(r2)
        up = _safe_unit(u2)

    fwd = _safe_unit(np.cross(right, up))
    right = _safe_unit(np.cross(up, fwd))
    up = _safe_unit(np.cross(fwd, right))

    right, up, fwd = _canonicalize_basis(right, up, fwd)
    return right, up, fwd, center

# ---------------------------------------
# חישוב Bind3Box: מציאת מימדי התיבה התוחמת לפי נקודות המגע המקסימליות
# ---------------------------------------

def build_bind3box_bbox_for_view(
    mesh: pv.PolyData,
    meta: dict,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any], Dict[str, Any]]:
    """
    מחשב את ה-Bind3Box לפי בסיס מבט MC:
    - מעביר נקודות למערכת הצירים של המצלמה.
    - מוצא קיצוניים גלובליים (extrema) לבניית ה-CaliperSet.
    - בונה תיבה תלת-ממדית ומחזיר אותה בקואורדינטות עולם.
    """
    if not isinstance(mesh, pv.PolyData):
        raise TypeError("build_bind3box_bbox_for_view: mesh must be pyvista.PolyData")
    if mesh.n_points == 0:
        raise ValueError("build_bind3box_bbox_for_view: empty mesh")

    right, up, fwd, center = _basis_from_meta(mesh, meta)

    P = np.asarray(mesh.points, dtype=float)
    Q = P - center[None, :]

    B = np.stack([right, fwd, up], axis=1)
    P_cam = Q @ B

    x = P_cam[:, 0]
    y = P_cam[:, 1]
    z = P_cam[:, 2]

    i_xmin = int(np.nanargmin(x))
    i_xmax = int(np.nanargmax(x))
    i_ymin = int(np.nanargmin(y))
    i_ymax = int(np.nanargmax(y))
    i_zmin = int(np.nanargmin(z))
    i_zmax = int(np.nanargmax(z))

    x1 = float(x[i_xmin]); z_at_x1 = float(z[i_xmin])
    x2 = float(x[i_xmax]); z_at_x2 = float(z[i_xmax])
    DD0 = CaliperSet(
        psCalw=np.array([x1, z_at_x1, x2, z_at_x2], float),
        psCalh=np.array([np.nan, np.nan, np.nan, np.nan], float),
    )

    y1 = float(y[i_ymin]); z_at_y1 = float(z[i_ymin])
    y2 = float(y[i_ymax]); z_at_y2 = float(z[i_ymax])

    z1 = float(z[i_zmin]); y_at_z1 = float(y[i_zmin])
    z2 = float(z[i_zmax]); y_at_z2 = float(y[i_zmax])

    DD1 = CaliperSet(
        psCalw=np.array([y1, z_at_y1, y2, z_at_y2], float),
        psCalh=np.array([y_at_z1, z1, y_at_z2, z2], float),
    )

    x_min, x_max = (min(x1, x2), max(x1, x2))
    y_min, y_max = (min(y1, y2), max(y1, y2))
    z_min, z_max = (min(z1, z2), max(z1, z2))

    r_cam = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        float,
    )
    c_cam = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2], float)
    vol = float((x_max - x_min) * (y_max - y_min) * (z_max - z_min))

    p6_idx = np.array([i_xmin, i_xmax, i_ymin, i_ymax, i_zmin, i_zmax], dtype=int)
    p6_cam = P_cam[p6_idx, :].copy()

    def back(pt_cam: np.ndarray) -> np.ndarray:
        return center + pt_cam[0] * right + pt_cam[1] * fwd + pt_cam[2] * up

    r_world = np.vstack([back(v) for v in r_cam])
    c_world = back(c_cam)
    p6_world = np.vstack([back(v) for v in p6_cam])

    p: Dict[str, Any] = {"center": c_world, "contacts6": p6_world}
    calipers: Dict[str, Any] = {"DD0": DD0, "DD1": DD1}
    return r_world, c_world, vol, p, calipers


def compute_bind3box_dimensions(calipers: Dict[str, Any]) -> Dict[str, float]:
    """
    מחשב את המימדים (רוחב, עומק, גובה) מתוך נתוני ה-CaliperSet.
    """
    DD0 = calipers.get("DD0", None)
    DD1 = calipers.get("DD1", None)
    if DD0 is None or DD1 is None:
        raise ValueError("compute_bind3box_dimensions: missing DD0/DD1 in calipers")

    x1 = float(DD0.psCalw[0]); x2 = float(DD0.psCalw[2])
    y1 = float(DD1.psCalw[0]); y2 = float(DD1.psCalw[2])
    z1 = float(DD1.psCalh[1]); z2 = float(DD1.psCalh[3])

    return {"dx": abs(x2 - x1), "dy": abs(y2 - y1), "dz": abs(z2 - z1)}


def build_view_caliper_box(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any], Dict[str, Any]]:
    """
    יצירת AABB פשוט לפי גבולות ה-mesh (Fallback).
    """
    if not isinstance(mesh, pv.PolyData):
        raise TypeError("build_view_caliper_box: mesh must be pyvista.PolyData")
    if mesh.n_points == 0:
        raise ValueError("build_view_caliper_box: empty mesh")

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    r = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )
    c = np.array([(xmax + xmin) * 0.5, (ymax + ymin) * 0.5, (zmax + zmin) * 0.5], dtype=float)
    vol = float(max(xmax - xmin, 0.0) * max(ymax - ymin, 0.0) * max(zmax - zmin, 0.0))
    return r, c, vol, {"center": c}, {}


# --------------------------------------------
# הקרנה למרחב המסך (Screen-space)
# --------------------------------------------

def project_points_to_pixels(
    P_world: np.ndarray,
    meta: dict,
    *,
    flip_v: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ממיר נקודות עולם לפיקסלים במסך על בסיס מטא-דאטה של המצלמה.
    אינו תלוי ב-Qt.
    """
    ps = float(meta["parallel_scale"])
    w, h = map(int, meta["window_size"])

    cx, cy, cz = map(float, meta.get("center", (0.0, 0.0, 0.0)))
    C = np.array([cx, cy, cz], float)

    if "right" not in meta or "up" not in meta:
        raise ValueError("project_points_to_pixels: meta must include 'right' and 'up'")

    right = np.asarray(meta["right"], float)
    up = np.asarray(meta["up"], float)
    right /= (np.linalg.norm(right) + 1e-12)
    up /= (np.linalg.norm(up) + 1e-12)

    if h <= 0 or w <= 0 or ps == 0.0:
        return np.zeros(0, float), np.zeros(0, float)

    aspect = (w / h) if h else 1.0
    x_half_world = ps * aspect

    D = np.asarray(P_world, float) - C[None, :]
    xw = D @ right
    yw = D @ up

    u = (xw / (2.0 * x_half_world) + 0.5) * w
    v = (yw / (2.0 * ps) + 0.5) * h
    if flip_v:
        v = h - v

    return u, v


def bbox_px_from_mesh_projection(
    mesh: pv.PolyData,
    meta: dict,
    *,
    sample: int = 12_000,
) -> tuple[int, int, int, int]:
    """
    מחשב תיבת תיחום (BBox) בפיקסלים עבור ההיטל של המודל על המסך.
    משתמש בדגימה דטרמיניסטית למניעת רעידות (flicker) בין רינדורים.
    """
    P = mesh.points
    n = int(P.shape[0])
    if n <= 0:
        return (0, 0, 0, 0)

    if n > sample:
        idx = np.linspace(0, n - 1, num=sample, dtype=int)
        P = P[idx]

    u, v = project_points_to_pixels(P, meta, flip_v=True)
    if u.size == 0 or v.size == 0:
        return (0, 0, 0, 0)

    u_min = int(np.floor(np.nanmin(u)))
    u_max = int(np.ceil(np.nanmax(u)))
    v_min = int(np.floor(np.nanmin(v)))
    v_max = int(np.ceil(np.nanmax(v)))

    w, h = map(int, meta["window_size"])
    u_min = max(0, min(u_min, w - 1))
    u_max = max(0, min(u_max, w - 1))
    v_min = max(0, min(v_min, h - 1))
    v_max = max(0, min(v_max, h - 1))

    return (u_min, u_max, v_min, v_max)