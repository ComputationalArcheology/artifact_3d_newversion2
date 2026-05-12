# src/artifact_app/processing/process_object_measurements.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist


# ------------------------------------------------------------
# מבני נתונים לשמירת תוצאות המדידה
# ------------------------------------------------------------

@dataclass
class CaliperSet:
    calw: float = 0.0
    psCalw: Optional[np.ndarray] = None
    calh: float = 0.0
    psCalh: Optional[np.ndarray] = None
    cordw: float = 0.0
    pCordw: Optional[np.ndarray] = None
    cordh: float = 0.0
    pCordh: Optional[np.ndarray] = None
    cordwhh: float = 0.0
    pCordwhh: Optional[np.ndarray] = None
    hcah: float = 0.0
    cord20p: float = 0.0
    pCord20p: Optional[np.ndarray] = None
    cord80p: float = 0.0
    pCord80p: Optional[np.ndarray] = None


@dataclass
class ProfileResult:
    horiz: str
    outline_uv: np.ndarray
    outline_source: str
    closed: bool
    matlab_data: Optional[CaliperSet] = None
    max_chord: float = 0.0
    chords_by_frac: Dict[float, Tuple[float, float, Any]] = None


@dataclass
class Bind3BoxResult:
    r_world: Optional[np.ndarray]
    center_world: Optional[np.ndarray]
    volume: float
    dims: Optional[Dict[str, float]]
    calipers: Optional[Dict[str, Any]]


@dataclass
class ProcessObjectMeasurements:
    n_points: int
    n_faces: int
    mesh_volume: float
    dx_world: float
    dy_world: float
    dz_world: float
    front_dx: float
    front_dy: float
    front_dz: float
    profile_xz: ProfileResult
    profile_yz: ProfileResult
    bind3box: Bind3BoxResult
    higher_touch_thickness: Optional[float]
    higher_touch_width: Optional[float]


# ------------------------------------------------------------
# פונקציית עזר לחישוב מרחקים (תואמת להתנהגות של MATLAB)
# ------------------------------------------------------------

def matlab_dist(w: np.ndarray) -> np.ndarray:
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    return cdist(w, w, metric='euclidean')


# ------------------------------------------------------------
# חישוב תיבה תוחמת (Bounding Box)
# ------------------------------------------------------------

def bind3box_matlab_style(DD0: CaliperSet, DD1: CaliperSet) -> Bind3BoxResult:
    if DD0.psCalw is None or DD1.psCalw is None or DD1.psCalh is None:
        return Bind3BoxResult(None, None, 0.0, {"dx": 0, "dy": 0, "dz": 0}, None)

    x1 = float(DD0.psCalw[0])
    x2 = float(DD0.psCalw[2])
    y1 = float(DD1.psCalw[0])
    y2 = float(DD1.psCalw[2])
    z1 = float(DD1.psCalh[1])
    z2 = float(DD1.psCalh[3])

    r_aligned = np.array([
        [x1, y1, z1], [x1, y1, z2], [x1, y2, z2], [x1, y2, z1],
        [x2, y2, z2], [x2, y2, z1], [x2, y1, z1], [x2, y1, z2]
    ])

    c_aligned = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2])

    dx_val = abs(x2 - x1)
    dy_val = abs(y2 - y1)
    dz_val = abs(z2 - z1)
    volume = dx_val * dy_val * dz_val

    dims = {
        "dx": dx_val,
        "dy": dy_val,
        "dz": dz_val
    }

    return Bind3BoxResult(
        r_world=r_aligned,
        center_world=c_aligned,
        volume=volume,
        dims=dims,
        calipers={"DD0": DD0, "DD1": DD1}
    )


# ------------------------------------------------------------
# איחוד קווי מתאר
# ------------------------------------------------------------

def unite_two_contours_matlab_style(w1: np.ndarray, w2: np.ndarray, dover_threshold: float) -> np.ndarray:
    if w1.shape[0] == 0:
        return w2
    if w2.shape[0] == 0:
        return w1

    w = np.vstack([w1, w2])
    wd = matlab_dist(w)
    swd1 = wd.shape[0]
    np.fill_diagonal(wd, np.nan)

    ip = []
    yp = []
    curr_idx = 0
    ip.append(curr_idx)
    yp.append(0.0)

    qq = 1
    while True:
        if qq >= swd1:
            break
        r = curr_idx
        row_dists = wd[r, :]
        if np.all(np.isnan(row_dists)):
            break
        next_idx = int(np.nanargmin(row_dists))
        Y = float(row_dists[next_idx])
        if Y >= 2.0:
            break
        wd[r, :] = np.nan
        wd[:, r] = np.nan
        curr_idx = next_idx
        qq += 1
        ip.append(curr_idx)
        yp.append(Y)

    yp_arr = np.array(yp, dtype=np.float64)
    vz = set(np.where(yp_arr == 0.0)[0].tolist())
    vt = np.where(yp_arr > dover_threshold)[0]

    if vt.size > 0:
        first_bad = int(vt[0])
        for k in range(first_bad, len(ip)):
            vz.add(k)

    final_indices = [idx for i, idx in enumerate(ip) if i not in vz]

    if not final_indices:
        return np.zeros((0, 2), dtype=np.float64)

    contour = w[final_indices].copy()

    if len(contour) > 2:
        contour = np.vstack([contour, contour[0]])

    if len(contour) > 3:
        xct = contour[:, 0]
        yct = contour[:, 1]
        dxct = np.diff(xct)
        yctav = (yct[:-1] + yct[1:]) / 2.0
        areax = np.sum(dxct * yctav)
        if areax < 0:
            contour = contour[::-1]

    return contour.astype(np.float64)


# ------------------------------------------------------------
# יצירת קו המתאר מהנקודות
# ------------------------------------------------------------

def draw_contour_matlab_style(v: np.ndarray, n_points_total: int, dmaxmin: np.ndarray) -> np.ndarray:
    if v is None or v.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    vmin = np.min(v, axis=0)
    vmax = np.max(v, axis=0)
    ivmin = np.argmin(v, axis=0)

    nslices = int(np.ceil(np.sqrt(v.shape[0]) / 1.5))
    if nslices < 10:
        nslices = 10

    pitch = (vmax[0] - vmin[0]) / nslices
    xin = vmin[0]
    ixin = int(ivmin[0])

    maxz = []
    minz = []

    for i in range(nslices):
        mask = (v[:, 0] > xin) & (v[:, 0] <= (xin + pitch))
        ix = np.where(mask)[0]

        if len(ix) < 2:
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
    mz = [ixin] + maxz_clean + minz_clean[::-1] + [ixin]

    pitch1 = (vmax[1] - vmin[1]) / nslices
    zin = vmin[1]
    izin = int(ivmin[1])

    maxx = []
    minx = []

    for i in range(nslices):
        mask = (v[:, 1] > zin) & (v[:, 1] <= (zin + pitch1))
        iz = np.where(mask)[0]

        if len(iz) < 2:
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
    mx = [izin] + maxx_clean + minx_clean[::-1] + [izin]

    w1 = v[mz].copy()
    w2 = v[mx].copy()
    dover = float(np.max(dmaxmin)) / 40.0

    contour = unite_two_contours_matlab_style(w1, w2, dover)
    return contour


# ------------------------------------------------------------
# מציאת רוחב המיתר (Chord) בגובה מסוים
# ------------------------------------------------------------

def cord_at_height_matlab(x: np.ndarray, y: np.ndarray, ymean: float) -> Tuple[float, List[float]]:
    n = len(x)
    if n < 3:
        return 0.0, []

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ymean = np.float64(ymean)

    diff_first = y[0] - ymean
    saa = -1 if diff_first < 0.0 else 1
    sa = saa
    icp = []

    for ii in range(1, n):
        diff_curr = y[ii] - ymean
        sb = -1 if diff_curr < 0.0 else 1
        if sb != sa:
            icp.append((ii - 1, ii))
            sa = sb

    if sa != saa:
        icp.append((n - 1, 0))

    if len(icp) != 2:
        return 0.0, []

    xx = []
    for ik in range(2):
        idx0, idx1 = icp[ik]
        p0_x = np.float64(x[idx0])
        p0_y = np.float64(y[idx0])
        p1_x = np.float64(x[idx1])
        p1_y = np.float64(y[idx1])

        numerator = (p1_x - p0_x) * (p0_y - ymean)
        denominator = p0_y - p1_y

        if abs(denominator) < 1e-15:
            dx_val = 0.0
        else:
            dx_val = numerator / denominator

        x_crossing = p0_x + dx_val
        xx.append(float(x_crossing))

    whh = abs(xx[0] - xx[1])
    return whh, xx


# ------------------------------------------------------------
# אלגוריתם מרכזי לחישוב המידות והמיתרים (תואם למקור)
# ------------------------------------------------------------

def width4a_matlab_algo(contour: np.ndarray, view_name: str, height_percent: float = 20.0) -> CaliperSet:
    D = CaliperSet()

    N = len(contour)
    if N < 5:
        return D

    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)

    # מדידות מעטפת בסיסיות (כחול)
    ixmax = int(np.argmax(x))
    xmax = float(x[ixmax])
    ixmin = int(np.argmin(x))
    xmin = float(x[ixmin])

    D.calw = xmax - xmin
    D.psCalw = np.array([xmin, y[ixmin], xmax, y[ixmax]], dtype=np.float64)

    iymax = int(np.argmax(y))
    ymax = float(y[iymax])
    iymin = int(np.argmin(y))
    ymin = float(y[iymin])

    D.calh = ymax - ymin
    D.psCalh = np.array([x[iymin], ymin, x[iymax], ymax], dtype=np.float64)

    hca = [y[ixmin] - ymin, y[ixmax] - ymin]
    D.hcah = float(max(hca))

    # חישוב רוחב מקסימלי (אדום אופקי)
    dyy = (ymax - ymin) / float(N) if N > 0 else 1.0
    L = np.zeros(N, dtype=np.float64)

    for i in range(N):
        curr_idx = (iymin + i) % N

        if curr_idx == iymax:
            break

        y_base = y[curr_idx]

        dy = 0.0
        r_indices = np.array([], dtype=int)
        loop_counter = 0

        while len(r_indices) < 3:
            loop_counter += 1
            if loop_counter > 10:
                break
            dy += dyy
            r_indices = np.where((y > y_base) & (y < (y_base + dy)))[0]

        if len(r_indices) == 0:
            continue

        dists = np.abs(x[r_indices] - x[curr_idx])
        sorted_order = np.argsort(dists)[::-1]
        sorted_r = r_indices[sorted_order]

        for k in sorted_r:
            ku = (k + 1) % N
            kd = (k - 1 + N) % N

            ey = y[k] - y_base
            if ey < 0:
                continue

            eyu = y[ku] - y_base
            eyd = y[kd] - y_base

            flg = ''
            if eyu <= 0:
                flg = 'u'
            elif eyd <= 0:
                flg = 'd'

            if flg:
                p0 = np.array([x[k], y[k]], dtype=np.float64)
                p1 = np.array([x[ku], y[ku]], dtype=np.float64) if flg == 'u' else np.array([x[kd], y[kd]],
                                                                                            dtype=np.float64)

                denom = p0[1] - p1[1]
                if abs(denom) > 1e-15:
                    dx = (p1[0] - p0[0]) * (p0[1] - y_base) / denom
                    x2 = p0[0] + dx
                    val = x2 - x[curr_idx]
                    if val > 0:
                        L[curr_idx] = val
                        break

    maxL = float(np.max(L))
    imaxL = int(np.argmax(L))
    D.cordw = maxL
    D.pCordw = np.array([x[imaxL], y[imaxL]], dtype=np.float64)

    # חישוב גובה מקסימלי (אדום אנכי)
    dxx = (xmax - xmin) / float(N) if N > 0 else 1.0
    M = np.zeros(N, dtype=np.float64)

    for i in range(N):
        curr_idx = (ixmin + i) % N

        if curr_idx == ixmax:
            break

        x_base = x[curr_idx]

        dx = 0.0
        r_indices = np.array([], dtype=int)
        loop_counter = 0

        while len(r_indices) < 3:
            loop_counter += 1
            if loop_counter > 10:
                break
            dx += dxx
            r_indices = np.where((x > x_base) & (x < (x_base + dx)))[0]

        if len(r_indices) == 0:
            continue

        dists = np.abs(y[r_indices] - y[curr_idx])
        sorted_order = np.argsort(dists)[::-1]
        sorted_r = r_indices[sorted_order]

        for k in sorted_r:
            ku = (k + 1) % N
            kd = (k - 1 + N) % N

            ex = x[k] - x_base
            if ex < 0:
                continue

            exu = x[ku] - x_base
            exd = x[kd] - x_base

            flg = ''
            if exu <= 0:
                flg = 'u'
            elif exd <= 0:
                flg = 'd'

            if flg:
                p0 = np.array([x[k], y[k]], dtype=np.float64)
                p1 = np.array([x[ku], y[ku]], dtype=np.float64) if flg == 'u' else np.array([x[kd], y[kd]],
                                                                                            dtype=np.float64)

                denom = p0[0] - p1[0]
                if abs(denom) > 1e-15:
                    dy = (p1[1] - p0[1]) * (p0[0] - x_base) / denom
                    y2 = p0[1] + dy
                    val = abs(y2 - y[curr_idx])
                    if val > 0:
                        M[curr_idx] = val
                        break

    maxM = float(np.max(M))
    imaxM = int(np.argmax(M))
    D.cordh = maxM
    D.pCordh = np.array([x[imaxM], y[imaxM]], dtype=np.float64)

    # חישוב רוחב בגבהים יחסיים (ירוק)
    ymean = (ymin + ymax) / 2.0
    whh, xx_50 = cord_at_height_matlab(x, y, ymean)
    D.cordwhh = whh
    if len(xx_50) >= 2:
        D.pCordwhh = np.array([min(xx_50), ymean], dtype=np.float64)

    frac = height_percent / 100.0

    y_low = ymin + (ymax - ymin) * frac
    w_low, xx_low = cord_at_height_matlab(x, y, y_low)
    D.cord20p = w_low
    if len(xx_low) >= 2:
        D.pCord20p = np.array([min(xx_low), y_low], dtype=np.float64)

    y_high = ymin + (ymax - ymin) * (1.0 - frac)
    w_high, xx_high = cord_at_height_matlab(x, y, y_high)
    D.cord80p = w_high
    if len(xx_high) >= 2:
        D.pCord80p = np.array([min(xx_high), y_high], dtype=np.float64)

    return D


# ------------------------------------------------------------
# הפונקציה הראשית שמרכזת את כל החישובים
# ------------------------------------------------------------

def compute_process_object_measurements(
        mesh: pv.PolyData,
        *,
        mc_meta: Dict[str, Any],
        view_size_xz: Tuple[int, int],
        view_size_yz: Tuple[int, int],
        height_percent: float = 20.0,
        **kwargs
) -> ProcessObjectMeasurements:
    pts = np.asarray(mesh.points).astype(np.float64)
    n_points_orig = mesh.n_points
    n_faces_orig = mesh.n_cells

    if pts.shape[0] == 0:
        empty_prof = ProfileResult("empty", np.zeros((0, 2)), "empty", False)
        empty_bbox = Bind3BoxResult(None, None, 0.0, {"dx": 0, "dy": 0, "dz": 0}, None)
        return ProcessObjectMeasurements(0, 0, 0, 0, 0, 0, 0, 0, 0, empty_prof, empty_prof, empty_bbox, 0, 0)

    bounds_min = np.min(pts, axis=0)
    bounds_max = np.max(pts, axis=0)
    dmaxmin = bounds_max - bounds_min

    uv_xz = pts[:, [0, 2]]
    cont_xz = draw_contour_matlab_style(uv_xz, pts.shape[0], dmaxmin[[0, 2]])

    uv_yz = pts[:, [1, 2]]
    cont_yz = draw_contour_matlab_style(uv_yz, pts.shape[0], dmaxmin[[1, 2]])

    D_xz = width4a_matlab_algo(cont_xz, "XZ_Thickness", height_percent=height_percent)
    D_yz = width4a_matlab_algo(cont_yz, "YZ_Width", height_percent=height_percent)

    try:
        bind3box = bind3box_matlab_style(D_xz, D_yz)
    except Exception:
        bind3box = Bind3BoxResult(None, None, 0.0, {"dx": 0, "dy": 0, "dz": 0}, None)

    def make_fracs(D: CaliperSet, vmin, vmax):
        res = {}
        ymean = (vmin + vmax) / 2
        x_start = D.pCordwhh[0] if D.pCordwhh is not None else 0
        res[0.5] = (ymean, D.cordwhh, (x_start, x_start + D.cordwhh))

        frac = height_percent / 100.0
        res[frac] = (vmin + (vmax - vmin) * frac, D.cord20p, None)
        res[1.0 - frac] = (vmin + (vmax - vmin) * (1.0 - frac), D.cord80p, None)
        return res

    prof_xz = ProfileResult(
        horiz="right",
        outline_uv=cont_xz,
        outline_source="MATLAB_EXACT",
        closed=True,
        matlab_data=D_xz,
        max_chord=D_xz.cordw,
        chords_by_frac=make_fracs(D_xz, D_xz.psCalh[1], D_xz.psCalh[3])
        if D_xz.psCalh is not None else {}
    )

    prof_yz = ProfileResult(
        horiz="fwd",
        outline_uv=cont_yz,
        outline_source="MATLAB_EXACT",
        closed=True,
        matlab_data=D_yz,
        max_chord=D_yz.cordw,
        chords_by_frac=make_fracs(D_yz, D_yz.psCalh[1], D_yz.psCalh[3])
        if D_yz.psCalh is not None else {}
    )

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