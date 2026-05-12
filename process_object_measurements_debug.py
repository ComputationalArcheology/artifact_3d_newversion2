# -*- coding: utf-8 -*-
# src/artifact_app/processing/process_object_measurements_debug.py
# DEBUG VERSION - prints detailed comparison info
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist

# Import the regular module
from artifact_app.processing.process_object_measurements import (
    CaliperSet,
    ProfileResult,
    Bind3BoxResult,
    ProcessObjectMeasurements,
    matlab_dist,
    bind3box_matlab_style,
    unite_two_contours_matlab_style,
    cord_at_height_matlab,
    width4a_matlab_algo,
)


def draw_contour_matlab_style_debug(v: np.ndarray, n_points_total: int, dmaxmin: np.ndarray,
                                    view_name: str = "") -> np.ndarray:
    """
    DEBUG version - prints detailed info for comparison with MATLAB
    """
    print(f"\n{'=' * 60}")
    print(f"[DEBUG] draw_contour_matlab_style - {view_name}")
    print(f"{'=' * 60}")

    if v is None or v.shape[0] < 3:
        print("[DEBUG] ERROR: v is None or too small")
        return np.zeros((0, 2), dtype=np.float64)

    print(f"[DEBUG] Input points: {v.shape[0]}")
    print(f"[DEBUG] n_points_total: {n_points_total}")

    # [vmin,ivmin]=min(v); vmax=max(v);
    vmin = np.min(v, axis=0)
    vmax = np.max(v, axis=0)
    ivmin = np.argmin(v, axis=0)

    print(f"[DEBUG] vmin = [{vmin[0]:.6f}, {vmin[1]:.6f}]")
    print(f"[DEBUG] vmax = [{vmax[0]:.6f}, {vmax[1]:.6f}]")
    print(f"[DEBUG] ivmin = [{ivmin[0]}, {ivmin[1]}]")

    # dmaxmin for this view
    dmaxmin_local = vmax - vmin
    print(f"[DEBUG] dmaxmin (local) = [{dmaxmin_local[0]:.6f}, {dmaxmin_local[1]:.6f}]")
    print(f"[DEBUG] dmaxmin (passed) = [{dmaxmin[0]:.6f}, {dmaxmin[1]:.6f}]")

    # nslices=ceil(sqrt(size(v,1))/1.5)
    nslices = int(np.ceil(np.sqrt(v.shape[0]) / 1.5))
    if nslices < 10:
        nslices = 10
    print(f"[DEBUG] nslices = {nslices}")

    # pitch=(vmax(1)-vmin(1))/nslices
    pitch = (vmax[0] - vmin[0]) / nslices
    print(f"[DEBUG] pitch (horizontal) = {pitch:.6f}")

    # --- Pass 1: Slice along Z, advance in X direction ---
    xin = vmin[0]
    ixin = int(ivmin[0])

    maxz = []
    minz = []
    egnor = []

    for i in range(nslices):
        mask = (v[:, 0] > xin) & (v[:, 0] <= (xin + pitch))
        ix = np.where(mask)[0]

        if len(ix) < 2:
            egnor.append(i)
            maxz.append(np.nan)
            minz.append(np.nan)
        else:
            slice_z = v[ix, 1]
            imiz = np.argmin(slice_z)
            imaz = np.argmax(slice_z)
            minz.append(ix[imiz])
            maxz.append(ix[imaz])

        xin += pitch

    maxz_clean = [int(x) for x in maxz if not np.isnan(x)]
    minz_clean = [int(x) for x in minz if not np.isnan(x)]

    print(f"[DEBUG] Pass 1: {len(egnor)} empty slices, {len(maxz_clean)} valid maxz, {len(minz_clean)} valid minz")

    mz = [ixin] + maxz_clean + minz_clean[::-1] + [ixin]
    print(f"[DEBUG] mz length = {len(mz)}")

    # --- Pass 2: Slice along X, advance in Z direction ---
    pitch1 = (vmax[1] - vmin[1]) / nslices
    print(f"[DEBUG] pitch1 (vertical) = {pitch1:.6f}")

    zin = vmin[1]
    izin = int(ivmin[1])

    maxx = []
    minx = []
    fgnor = []

    for i in range(nslices):
        mask = (v[:, 1] > zin) & (v[:, 1] <= (zin + pitch1))
        iz = np.where(mask)[0]

        if len(iz) < 2:
            fgnor.append(i)
            maxx.append(np.nan)
            minx.append(np.nan)
        else:
            slice_x = v[iz, 0]
            imix = np.argmin(slice_x)
            imax = np.argmax(slice_x)
            minx.append(iz[imix])
            maxx.append(iz[imax])

        zin += pitch1

    maxx_clean = [int(x) for x in maxx if not np.isnan(x)]
    minx_clean = [int(x) for x in minx if not np.isnan(x)]

    print(f"[DEBUG] Pass 2: {len(fgnor)} empty slices, {len(maxx_clean)} valid maxx, {len(minx_clean)} valid minx")

    mx = [izin] + maxx_clean + minx_clean[::-1] + [izin]
    print(f"[DEBUG] mx length = {len(mx)}")

    # --- Build w1, w2 for UniteTwoContours ---
    w1 = v[mz].copy()
    w2 = v[mx].copy()

    print(f"[DEBUG] w1 shape = {w1.shape}")
    print(f"[DEBUG] w2 shape = {w2.shape}")

    # dover=max(dmaxmin)/40
    dover = float(np.max(dmaxmin)) / 40.0
    print(f"[DEBUG] dover = {dover:.6f}")

    # Call UniteTwoContours
    contour = unite_two_contours_matlab_style(w1, w2, dover)

    print(f"[DEBUG] Final contour shape = {contour.shape}")
    if contour.shape[0] > 0:
        print(
            f"[DEBUG] Contour bounds: x=[{contour[:, 0].min():.4f}, {contour[:, 0].max():.4f}], z=[{contour[:, 1].min():.4f}, {contour[:, 1].max():.4f}]")
        print(f"[DEBUG] First 3 points: {contour[:3]}")
        print(f"[DEBUG] Last 3 points: {contour[-3:]}")

    return contour


def compute_process_object_measurements_debug(
        mesh: pv.PolyData,
        *,
        mc_meta: Dict[str, Any],
        view_size_xz: tuple[int, int],
        view_size_yz: tuple[int, int],
        **kwargs
) -> ProcessObjectMeasurements:
    """
    DEBUG version of compute_process_object_measurements
    """
    print("\n" + "=" * 80)
    print("[DEBUG] compute_process_object_measurements_debug")
    print("=" * 80)

    pts = np.asarray(mesh.points).astype(np.float64)
    n_points_orig = mesh.n_points
    n_faces_orig = mesh.n_cells

    print(f"[DEBUG] Mesh: {n_points_orig} points, {n_faces_orig} faces")

    if pts.shape[0] == 0:
        print("[DEBUG] ERROR: Empty mesh!")
        empty_prof = ProfileResult("empty", np.zeros((0, 2)), "empty", False)
        empty_bbox = Bind3BoxResult(None, None, 0.0, {"dx": 0, "dy": 0, "dz": 0}, None)
        return ProcessObjectMeasurements(0, 0, 0, 0, 0, 0, 0, 0, 0, empty_prof, empty_prof, empty_bbox, 0, 0)

    # Bounds
    bounds_min = np.min(pts, axis=0)
    bounds_max = np.max(pts, axis=0)
    dmaxmin = bounds_max - bounds_min

    print(f"[DEBUG] Mesh bounds:")
    print(f"[DEBUG]   X: [{bounds_min[0]:.6f}, {bounds_max[0]:.6f}] -> dx = {dmaxmin[0]:.6f}")
    print(f"[DEBUG]   Y: [{bounds_min[1]:.6f}, {bounds_max[1]:.6f}] -> dy = {dmaxmin[1]:.6f}")
    print(f"[DEBUG]   Z: [{bounds_min[2]:.6f}, {bounds_max[2]:.6f}] -> dz = {dmaxmin[2]:.6f}")

    # --- Generate Contours ---
    uv_xz = pts[:, [0, 2]]  # [x, z]
    cont_xz = draw_contour_matlab_style_debug(uv_xz, pts.shape[0], dmaxmin[[0, 2]], "XZ (Thickness)")

    uv_yz = pts[:, [1, 2]]  # [y, z]
    cont_yz = draw_contour_matlab_style_debug(uv_yz, pts.shape[0], dmaxmin[[1, 2]], "YZ (Width)")

    # --- Measure Profiles ---
    print("\n[DEBUG] Measuring XZ profile...")
    D_xz = width4a_matlab_algo(cont_xz, "XZ_Thickness")
    print(
        f"[DEBUG] XZ results: calw={D_xz.calw:.4f}, calh={D_xz.calh:.4f}, cordw={D_xz.cordw:.4f}, cordwhh={D_xz.cordwhh:.4f}")

    print("\n[DEBUG] Measuring YZ profile...")
    D_yz = width4a_matlab_algo(cont_yz, "YZ_Width")
    print(
        f"[DEBUG] YZ results: calw={D_yz.calw:.4f}, calh={D_yz.calh:.4f}, cordw={D_yz.cordw:.4f}, cordwhh={D_yz.cordwhh:.4f}")

    # --- Bind3Box ---
    try:
        bind3box = bind3box_matlab_style(D_xz, D_yz)
        print(
            f"\n[DEBUG] Bind3Box dims: dx={bind3box.dims['dx']:.4f}, dy={bind3box.dims['dy']:.4f}, dz={bind3box.dims['dz']:.4f}")
    except Exception as e:
        print(f"[DEBUG] Bind3Box error: {e}")
        bind3box = Bind3BoxResult(None, None, 0.0, {"dx": 0, "dy": 0, "dz": 0}, None)

    # --- Pack Results ---
    def make_fracs(D: CaliperSet, vmin, vmax):
        res = {}
        ymean = (vmin + vmax) / 2
        x_start = D.pCordwhh[0] if D.pCordwhh is not None else 0
        res[0.5] = (ymean, D.cordwhh, (x_start, x_start + D.cordwhh))
        res[0.2] = (vmin + (vmax - vmin) / 5, D.cord20p, None)
        res[0.8] = (vmin + (vmax - vmin) * 4 / 5, D.cord80p, None)
        return res

    prof_xz = ProfileResult(
        horiz="right",
        outline_uv=cont_xz,
        outline_source="MATLAB_DEBUG",
        closed=True,
        matlab_data=D_xz,
        max_chord=D_xz.cordw,
        chords_by_frac=make_fracs(D_xz, D_xz.psCalh[1], D_xz.psCalh[3])
        if D_xz.psCalh is not None else {}
    )

    prof_yz = ProfileResult(
        horiz="fwd",
        outline_uv=cont_yz,
        outline_source="MATLAB_DEBUG",
        closed=True,
        matlab_data=D_yz,
        max_chord=D_yz.cordw,
        chords_by_frac=make_fracs(D_yz, D_yz.psCalh[1], D_yz.psCalh[3])
        if D_yz.psCalh is not None else {}
    )

    print("\n" + "=" * 80)
    print("[DEBUG] SUMMARY - Compare with MATLAB:")
    print("=" * 80)
    print(f"  Length (Height/calh): {bind3box.dims['dz']:.4f}")
    print(f"  Width (calw XZ):      {bind3box.dims['dx']:.4f}")
    print(f"  Thickness (calw YZ):  {bind3box.dims['dy']:.4f}")
    print(f"  XZ Max Cord:          {D_xz.cordw:.4f}")
    print(f"  YZ Max Cord:          {D_yz.cordw:.4f}")
    print(f"  XZ @ 50%:             {D_xz.cordwhh:.4f}")
    print(f"  YZ @ 50%:             {D_yz.cordwhh:.4f}")
    print(f"  XZ @ 20%:             {D_xz.cord20p:.4f}")
    print(f"  YZ @ 20%:             {D_yz.cord20p:.4f}")
    print(f"  XZ @ 80%:             {D_xz.cord80p:.4f}")
    print(f"  YZ @ 80%:             {D_yz.cord80p:.4f}")
    print("=" * 80 + "\n")

    return ProcessObjectMeasurements(
        n_points=n_points_orig,
        n_faces=n_faces_orig,
        mesh_volume=mesh.volume,
        dx_world=bind3box.dims["dx"],
        dy_world=bind3box.dims["dy"],
        dz_world=bind3box.dims["dz"],
        front_dx=bind3box.dims["dx"],
        front_dy=bind3box.dims["dy"],
        front_dz=bind3box.dims["dz"],
        profile_xz=prof_xz,
        profile_yz=prof_yz,
        bind3box=bind3box,
        higher_touch_thickness=D_yz.hcah,
        higher_touch_width=D_xz.hcah
    )