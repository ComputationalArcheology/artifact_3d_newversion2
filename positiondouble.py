# -*- coding: utf-8 -*-
# src/artifact_app/processing/position.py
from __future__ import annotations

import numpy as np
import pyvista as pv
from artifact_app.processing.center_of_mass import compute_area_weighted_centroid

# סיבוב 180° סביב ציר X (עבור היפוך ציר Z, תואם לוגיקת Ult3Pose)
RX_PI = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]], dtype=float)

# סיבוב 90° סביב ציר Z (כדי להתאים את קונבנציית המבט של MATLAB לזו של PyVista)
RZ_90 = np.array([[0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)


def _compute_normal_tensor(m: pv.PolyData) -> tuple[np.ndarray, float]:
    """
    חישוב טנזור נורמלים משוקללי שטח (זהה לחישוב ב-MATLAB).
    """
    tri = m if m.is_all_triangles else m.triangulate(inplace=False)
    tri = tri.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    tri = tri.compute_cell_sizes(length=False, area=True, volume=False)

    vn = tri.cell_normals.astype(np.float64)
    tra = tri.cell_data["Area"].astype(np.float64)
    SA = float(np.sum(tra)) or 1.0

    Sxx = np.sum(tra * (vn[:, 0] ** 2))
    Syy = np.sum(tra * (vn[:, 1] ** 2))
    Szz = np.sum(tra * (vn[:, 2] ** 2))
    Sxy = np.sum(tra * (vn[:, 0] * vn[:, 1]))
    Syz = np.sum(tra * (vn[:, 1] * vn[:, 2]))
    Szx = np.sum(tra * (vn[:, 2] * vn[:, 0]))

    M = np.array([
        [Sxx, Sxy, Szx],
        [Sxy, Syy, Syz],
        [Szx, Syz, Szz]
    ], dtype=np.float64) / SA

    return M, SA


def _norm2_posing(M: np.ndarray) -> np.ndarray:
    """
    מחשב את מטריצת היישור (Ttr) על בסיס וקטורים עצמיים, תוך תיקון כיווני הצירים
    כדי להבטיח התאמה מלאה למערכת הצירים ב-MATLAB.
    """
    vals, V = np.linalg.eig(M)
    vals = np.real(vals)
    V = np.real(V)

    colmax = int(np.argmax(vals))
    colmin = int(np.argmin(vals))

    xax = V[:, colmax].copy()
    zax = V[:, colmin].copy()

    # תיקון סימן לציר Z: רכיב ה-Z חייב להיות חיובי כדי שציר האורך יפנה כלפי מעלה
    if zax[2] < 0:
        zax = -zax

    yax = np.cross(zax, xax)

    xax = xax / (np.linalg.norm(xax) + 1e-12)
    yax = yax / (np.linalg.norm(yax) + 1e-12)
    zax = zax / (np.linalg.norm(zax) + 1e-12)

    Ttr = np.column_stack([xax, yax, zax])

    # שמירה על מערכת צירים ימנית (Right-handed)
    if np.linalg.det(Ttr) < 0:
        yax = -yax
        Ttr = np.column_stack([xax, yax, zax])

    return Ttr


def align_mesh(mesh: pv.PolyData, method: str = "matlab"):
    """
    תהליך היישור המלא של האובייקט: מירכוז למרכז המסה, חישוב מטריצת יישור,
    והחלת סיבובים נדרשים לתאימות גרפית ולוגית.
    """
    aligned = mesh.copy(deep=True)

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

    P_centered = np.asarray(aligned.points, dtype=np.float64)

    M, _ = _compute_normal_tensor(aligned)
    Ttr = _norm2_posing(M)

    Tuzy = Ttr.T

    pts_rotated = (Tuzy @ P_centered.T).T

    # היפוך ציר Z אם צריך (שחזור לוגיקת Ult3Pose מ-MATLAB)
    z_vals = pts_rotated[:, 2]
    midz = (z_vals.max() + z_vals.min()) / 2.0

    if midz < 0:
        pts_rotated = (RX_PI @ pts_rotated.T).T
        Tuzy = RX_PI @ Tuzy
        print("[Align] Applied Z-flip (midz < 0)")

    # סיבוב לתאימות עם כיוון המצלמה של PyVista
    pts_rotated = (RZ_90 @ pts_rotated.T).T
    Tuzy = RZ_90 @ Tuzy

    aligned.points = pts_rotated

    print(f"[Align] Bounds after alignment:")
    print(
        f"  X: [{pts_rotated[:, 0].min():.2f}, {pts_rotated[:, 0].max():.2f}] -> {pts_rotated[:, 0].max() - pts_rotated[:, 0].min():.2f}")
    print(
        f"  Y: [{pts_rotated[:, 1].min():.2f}, {pts_rotated[:, 1].max():.2f}] -> {pts_rotated[:, 1].max() - pts_rotated[:, 1].min():.2f}")
    print(
        f"  Z: [{pts_rotated[:, 2].min():.2f}, {pts_rotated[:, 2].max():.2f}] -> {pts_rotated[:, 2].max() - pts_rotated[:, 2].min():.2f}")

    test_com, _ = compute_area_weighted_centroid(aligned)
    print(f"DEBUG - Center of Mass after alignment: ({test_com[0]:.4f}, {test_com[1]:.4f}, {test_com[2]:.4f})")

    return aligned, Tuzy