# -*- coding: utf-8 -*-
# src/artifact_app/processing/position.py
from __future__ import annotations

import numpy as np
import pyvista as pv
from artifact_app.processing.center_of_mass import compute_area_weighted_centroid


def _compute_normal_tensor(mesh: pv.PolyData) -> tuple[np.ndarray, float]:
    """שחזור מדויק של חישוב הנורמלים לפי UzyPosCm_for_GUI.m"""
    tri = mesh if mesh.is_all_triangles else mesh.triangulate(inplace=False)
    tri = tri.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    tri = tri.compute_cell_sizes(length=False, area=True, volume=False)

    fn = tri.cell_normals.astype(np.float64)
    tra = tri.cell_data["Area"].astype(np.float64)
    SA = float(np.sum(tra)) or 1.0

    Sxx = np.sum(tra * (fn[:, 0] ** 2))
    Syy = np.sum(tra * (fn[:, 1] ** 2))
    Szz = np.sum(tra * (fn[:, 2] ** 2))
    Sxy = np.sum(tra * (fn[:, 0] * fn[:, 1]))
    Syz = np.sum(tra * (fn[:, 1] * fn[:, 2]))
    Szx = np.sum(tra * (fn[:, 2] * fn[:, 0]))

    M = np.array([
        [Sxx, Sxy, Szx],
        [Sxy, Syy, Syz],
        [Szx, Syz, Szz]
    ], dtype=np.float64) / SA

    return M, SA


def _norm2_posing(M: np.ndarray) -> np.ndarray:
    """שחזור אחד-לאחד של הפונקציה Norm2Posing_for_GUI.m מהמקור"""
    vals, V = np.linalg.eigh(M)

    colmax = int(np.argmax(vals))
    colmin = int(np.argmin(vals))

    zax = V[:, colmin].copy()
    xax = V[:, colmax].copy()
    tyax = np.cross(zax, xax)

    # שחזור הלוגיקה המקורית של מטלב למציאת colav (1-based to 0-based)
    csc = sorted([colmin, colmax])
    if csc[0] == 1:
        colav = 0
    elif csc[1] == 1:
        colav = 2
    else:
        colav = 1

    yax_temp = V[:, colav]
    # בדיקת אורתוגונליות בדיוק כמו במטלב
    if np.any(np.abs(yax_temp - tyax) > 8 * np.finfo(float).eps):
        yax = -yax_temp
    else:
        yax = yax_temp.copy()

    Ttr = np.column_stack([xax, yax, zax])
    return Ttr


def align_mesh(mesh: pv.PolyData, method: str = "matlab"):
    aligned = mesh.copy(deep=True)

    # --- 1. מירכוז המסה (שחזור ההזזה ב- qfun_01_2011) ---
    pts = np.asarray(aligned.points, dtype=np.float64)
    mv = pts.mean(axis=0)
    pts = pts - mv
    aligned.points = pts

    iteration = 0
    max_iterations = 3
    while iteration < max_iterations:
        try:
            com, vol = compute_area_weighted_centroid(aligned)
        except Exception:
            break
        if vol == 0:
            aligned.points -= com
            break
        moments = com * vol
        if np.max(np.abs(moments)) <= 1.0:
            break
        aligned.points -= com
        iteration += 1

    vcm = np.asarray(aligned.points, dtype=np.float64)

    # --- 2. חישוב הצירים והמטריצה ---
    M, _ = _compute_normal_tensor(aligned)
    Ttr = _norm2_posing(M)

    # --- 3. שחזור מדויק של Ult2Pose_for_GUI ---
    InvTrr = Ttr.T
    vct = (InvTrr @ vcm.T).T

    # מבחן האורך (midz) - בדיוק כמו במטלב (מוודא שהבסיס הכבד למטה)
    midz = (vct[:, 2].max() + vct[:, 2].min()) / 2.0
    if midz < 0:
        RX_180 = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]], dtype=np.float64)
        InvTrr = RX_180 @ InvTrr
        vct = (InvTrr @ vcm.T).T
        print("[Align] Applied MATLAB midz<0 check (Heavy base down)")

    Tuzy = InvTrr

    # --- 4. מבחן השולחן החסר! (Ventral / Dorsal) ---
    # ציר X הוא העובי (Thickness). אנו מוודאים מתמטית שהצד הקמור תמיד ב- +X
    # כדי שלא נישען על מזל של פונקציית eig. כשהצד הקמור ב- +X, הצד השטוח יונח בבטחה על השולחן.
    midx = (vct[:, 0].max() + vct[:, 0].min()) / 2.0
    if midx < 0:
        # היפוך סביב ציר Z שומר על האורך (Z) אבל הופך את העובי (X) והרוחב (Y)
        RZ_180 = np.array([[-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        Tuzy = RZ_180 @ Tuzy
        print("[Align] Applied X-flip to strictly enforce convex side UP")

    # --- 5. התאמה למצלמה של PyVista (מבט רחב) ---
    # כדי שחלון המבט הראשי (MC) שלך יראה את מישור הרוחב במקום העובי כמו במטלב,
    # אנו מסובבים 90 מעלות. כעת הצד הקמור יפנה ישירות למצלמה!
    RZ_90 = np.array([[0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    Tuzy = RZ_90 @ Tuzy
    pts_rotated = (Tuzy @ vcm.T).T

    aligned.points = pts_rotated
    return aligned, Tuzy