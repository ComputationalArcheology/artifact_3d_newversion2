# src/artifact_app/viewer/view_matlab_style.py
"""
גרסה V6 - תיקון המיקום של משבצת 7 (CUT_RESULT) לפינה השמאלית-תחתונה המדויקת!
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QWidget, QLabel

from artifact_app.gui.aspect_label import AspectPixmapLabel
from artifact_app.viewer.views_spec import get_views_spec
from artifact_app.viewer.main_view import set_view_azel, compute_view_dirs_from_azel


# ============================================================
#  A) פונקציות עזר לפריסה (Layout) בסגנון MATLAB
# ============================================================
@dataclass
class NRect:
    x: float
    y: float
    w: float
    h: float


def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def _subplot33_positions_matlab_like(
        *,
        L: float = 0.0,
        R: float = 0.0,
        B: float = 0.0,
        T: float = 0.0,
        gx: float = 0.0,
        gy: float = 0.0,
) -> dict[int, NRect]:
    cw = (1.0 - L - R - 2.0 * gx) / 3.0
    ch = (1.0 - B - T - 2.0 * gy) / 3.0

    def cell(row_from_bottom: int, col: int) -> NRect:
        x = L + col * (cw + gx)
        y = B + row_from_bottom * (ch + gy)
        return NRect(x, y, cw, ch)

    return {
        1: cell(2, 0),
        4: cell(1, 0),
        5: cell(1, 1),
        6: cell(1, 2),
        7: cell(0, 0),  # <--- משבצת 7 התחתונה
        8: cell(0, 1),
        9: cell(0, 2),
    }


def compute_matlab_5rects_normalized(dx: float, dy: float, dz: float) -> dict[str, NRect]:
    """
    מחשב את המיקומים והגדלים של 5 המבטים, מנורמל למסך, כך שיישמר יחס הפרופורציה בין המבטים לפי ה-Bounding Box.
    """
    # התיקון היחיד: נותנים לכל משבצת את היחס הפיזיקלי המדויק שהמצלמה רואה (גובה חלקי רוחב)
    r = {
        1: _safe_div(dy, dx, 1.0),  # BR (למטה): רואה גובה Y, רוחב X
        4: _safe_div(dz, dx, 1.0),  # MC (ראשי שמאלי): רואה גובה Z, רוחב X
        5: _safe_div(dz, dy, 1.0),  # ML (צד אמצעי): רואה גובה Z, רוחב Y
        6: _safe_div(dz, dx, 1.0),  # MR (צד ימני): רואה גובה Z, רוחב X
        9: _safe_div(dy, dx, 1.0),  # TL (למעלה): רואה גובה Y, רוחב X
    }

    pos = _subplot33_positions_matlab_like()
    qos = {i: NRect(pos[i].x, pos[i].y, pos[i].w, pos[i].h) for i in (1, 4, 5, 6, 9)}

    for i, p in qos.items():
        rt = _safe_div(p.h, p.w, 1.0)
        if rt > r[i]:
            qos[i].h = p.w * r[i]
        else:
            qos[i].w = _safe_div(p.h, r[i], p.w)

    refv5 = qos[5].h
    for i in (4, 6):
        q = qos[i]
        rat = _safe_div(refv5, q.h, 1.0)
        qos[i] = NRect(q.x, q.y, q.w * rat, q.h * rat)

    refh4 = qos[4].w
    q1 = qos[1]
    rat = _safe_div(refh4, q1.w, 1.0)
    qos[1] = NRect(q1.x, q1.y, q1.w * rat, q1.h * rat)

    refh6 = qos[6].w
    q9 = qos[9]
    rat = _safe_div(refh6, q9.w, 1.0)
    qos[9] = NRect(q9.x, q9.y, q9.w * rat, q9.h * rat)

    e1 = pos[4].x
    e2 = pos[6].x + pos[6].w
    q1x = qos[4].x
    q2x = q1x + qos[4].w + qos[5].w + qos[6].w
    sch = _safe_div((q2x - q1x), (e2 - e1), 1.0)

    e3 = pos[9].y
    e4 = pos[1].y + pos[1].h
    q3y = qos[9].y
    q4y = qos[9].y + qos[9].h + qos[6].h + qos[1].h
    scv = _safe_div((q4y - q3y), (e4 - e3), 1.0)

    sc = max(sch, scv)

    ros = {i: NRect(0, 0, 0, 0) for i in (1, 4, 5, 6, 9)}
    ros[1].x = (1.0 - (q2x - q1x) / sc) / 2.0
    ros[1].w = qos[1].w / sc
    ros[1].h = qos[1].h / sc

    ros[4].x = ros[1].x
    ros[4].w = qos[4].w / sc
    ros[4].h = qos[4].h / sc

    ros[5].x = ros[4].x + ros[4].w
    ros[5].w = qos[5].w / sc
    ros[5].h = qos[5].h / sc

    ros[6].x = ros[5].x + ros[5].w
    ros[6].w = qos[6].w / sc
    ros[6].h = qos[6].h / sc

    ros[9].x = ros[6].x
    ros[9].w = qos[9].w / sc
    ros[9].h = qos[9].h / sc

    spanh = (q2x - q1x) / sc
    spanv = (q4y - q3y) / sc

    if spanh > spanv:
        mL = ros[4].x
        mR = 1.0 - mL - spanh
        seph = (mL + mR) / 4.0

        ros[4].x -= seph
        ros[1].x -= seph
        ros[6].x += seph
        ros[9].x += seph

        tbt = (1.0 - spanv - 2.0 * seph) / 2.0
        ros[9].y = tbt
        ros[4].y = ros[9].y + ros[9].h + seph
        ros[5].y = ros[4].y
        ros[6].y = ros[4].y
        ros[1].y = 1.0 - tbt - ros[1].h
    else:
        mB = ros[9].y
        mT = 1.0 - mB - spanv
        sepv = (mB + mT) / 4.0

        ros[9].y = sepv
        ros[4].y = ros[9].y + ros[9].h + sepv
        ros[5].y = ros[4].y
        ros[6].y = ros[4].y
        ros[1].y = ros[6].y + ros[6].h + sepv

        tbt = (1.0 - spanh - 2.0 * sepv) / 2.0
        ros[4].x = tbt
        ros[1].x = tbt
        ros[9].x += sepv
        ros[6].x += sepv

    return {
        "BR": ros[1],
        "MC": ros[4],
        "ML": ros[5],
        "MR": ros[6],
        "TL": ros[9],
        "CUT_RESULT": NRect(ros[4].x, ros[9].y, ros[4].w, ros[9].h)
    }
class MatlabFiveViewsCanvas(QWidget):
    KEYS = ("TL", "TITLE", "CUT_UP", "ML", "MC", "MR", "CUT_DN", "SCALE", "BR", "CUT_RESULT")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels: dict[str, QWidget] = {}

        for k in self.KEYS:
            if k in ("SCALE", "CUT_RESULT"):
                lbl = QLabel(self)
            else:
                lbl = AspectPixmapLabel(self)

            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: transparent; border: none;")
            lbl.setObjectName(k)
            self.labels[k] = lbl
            lbl.show()

    def get_view_sizes(self) -> dict[str, tuple[int, int]]:
        render_keys = ("TL", "ML", "MC", "MR", "BR")
        out: dict[str, tuple[int, int]] = {}
        for k in render_keys:
            lbl = self.labels.get(k)
            if lbl is None:
                continue
            out[k] = (max(1, lbl.width()), max(1, lbl.height()))
        return out

    def set_bbox(self, dx: float, dy: float, dz: float) -> None:
        self._bbox_dx, self._bbox_dy, self._bbox_dz = float(dx), float(dy), float(dz)
        self._layout_matlab_like()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_matlab_like()

    def _layout_matlab_like(self) -> None:
        W, H = self.width(), self.height()
        pad = 0
        inner_w = max(1, W - 2 * pad)
        inner_h = max(1, H - 2 * pad)

        dx = getattr(self, "_bbox_dx", None)
        dy = getattr(self, "_bbox_dy", None)
        dz = getattr(self, "_bbox_dz", None)

        def to_qrect(nr: NRect) -> QRect:
            x = pad + int(round(nr.x * inner_w))
            w = int(round(nr.w * inner_w))
            y_top = 1.0 - (nr.y + nr.h)
            y = pad + int(round(y_top * inner_h))
            h = int(round(nr.h * inner_h))
            return QRect(x, y, max(1, w), max(1, h))

        if not (dx and dy and dz):
            pos = _subplot33_positions_matlab_like()
            rects = {}
            for idx, key in [(9, "TL"), (5, "ML"), (4, "MC"), (6, "MR"), (1, "BR"), (8, "SCALE"), (7, "CUT_RESULT")]:
                rects[key] = to_qrect(pos[idx])
        else:
            nrects = compute_matlab_5rects_normalized(float(dx), float(dy), float(dz))
            rects = {k: to_qrect(v) for k, v in nrects.items()}

            mid = nrects["ML"]
            bottom_key = min(("TL", "BR"), key=lambda k: float(nrects[k].y))
            bottom = nrects[bottom_key]
            rects["SCALE"] = to_qrect(NRect(mid.x, bottom.y, mid.w, bottom.h))

        for key, rect in rects.items():
            lbl = self.labels.get(key)
            if lbl is not None:
                lbl.setGeometry(rect)

        if "SCALE" in self.labels:
            self.labels["SCALE"].raise_()
            self.labels["SCALE"].show()


# ============================================================
#  B) Rendering - שימוש במרווח גלובלי להשארת מקום לסרגלים
# ============================================================

GLOBAL_RULER_MARGIN = 1.10

def _np_to_qpixmap_rgb(img_rgb: np.ndarray) -> QPixmap:
    a = np.asarray(img_rgb)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] > 3:
        a = a[..., :3]
    a = np.ascontiguousarray(a, dtype=np.uint8)
    h, w, _ = a.shape
    qimg = QImage(a.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

def _basis_from_camera(pl: pv.Plotter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = pl.camera
    pos = np.asarray(cam.position, float)
    focal = np.asarray(cam.focal_point, float)

    view_dir = focal - pos
    view_dir /= (np.linalg.norm(view_dir) + 1e-12)

    up = np.asarray(cam.up, float)
    up = up - view_dir * float(np.dot(up, view_dir))
    up /= (np.linalg.norm(up) + 1e-12)

    right = np.cross(view_dir, up)
    right /= (np.linalg.norm(right) + 1e-12)

    up = np.cross(right, view_dir)
    up /= (np.linalg.norm(up) + 1e-12)

    fwd = np.cross(right, up)
    fwd /= (np.linalg.norm(fwd) + 1e-12)

    return right, up, fwd

def _basis_from_azel_roll(az: float, el: float, roll: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, up, fwd = compute_view_dirs_from_azel(float(az), float(el))
    right = np.asarray(right, float)
    up = np.asarray(up, float)
    fwd = np.asarray(fwd, float)

    right /= (np.linalg.norm(right) + 1e-12)
    up /= (np.linalg.norm(up) + 1e-12)
    fwd /= (np.linalg.norm(fwd) + 1e-12)

    rr = np.deg2rad(float(roll))
    if rr != 0.0:
        c, s = np.cos(rr), np.sin(rr)
        r2 = c * right + s * up
        u2 = -s * right + c * up
        right = r2 / (np.linalg.norm(r2) + 1e-12)
        up = u2 / (np.linalg.norm(u2) + 1e-12)

    right = np.cross(up, fwd)
    right /= (np.linalg.norm(right) + 1e-12)
    up = np.cross(fwd, right)
    up /= (np.linalg.norm(up) + 1e-12)

    return right, up, fwd

def _project_spans_exact(
        pts: np.ndarray,
        center: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    d = pts - center[None, :]
    r = d @ right
    u = d @ up

    r_min = float(r.min())
    r_max = float(r.max())
    u_min = float(u.min())
    u_max = float(u.max())

    h_span = r_max - r_min
    v_span = u_max - u_min

    return r_min, r_max, u_min, u_max, h_span, v_span

def _render_single_view_img_and_meta(
        mesh: pv.PolyData,
        *,
        view_name: str,
        forced_parallel_scale: float | None = None,
        window_size: tuple[int, int],
        background: str,
        color: str,
        az: float,
        el: float,
        roll: float,
        zoom_fact: float,
        multi_samples: int,
        dist_scale: float = 1.9,
        points_world: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    w_r, h_r = max(1, int(window_size[0])), max(1, int(window_size[1]))

    pl = pv.Plotter(off_screen=True, window_size=(w_r, h_r), lighting='none')

    light_front = pv.Light(light_type='headlight')
    light_front.intensity = 0.8
    pl.add_light(light_front)

    light_top = pv.Light(position=(10, 10, 10), light_type='camera light')
    light_top.intensity = 0.5
    pl.add_light(light_top)

    if multi_samples > 0:
        try:
            pl.enable_anti_aliasing("msaa", multi_samples=multi_samples)
        except Exception:
            pass

    pl.set_background(background if background else "white")

    try:
        mesh_draw = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            splitting=False,
            auto_orient_normals=True,
            inplace=False,
        )
    except Exception:
        mesh_draw = mesh

    pl.add_mesh(
        mesh_draw,
        color=color,
        smooth_shading=True,
        ambient=0.1,
        diffuse=0.7,
        specular=0.1,
        specular_power=30,
        show_edges=False
    )

    xmin, xmax, ymin, ymax, zmin, zmax = mesh_draw.bounds
    center = np.array(
        [(xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0],
        float,
    )

    radius = 0.5 * float(max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-12))
    r_used = max(radius, 1e-6) * float(dist_scale)

    set_view_azel(
        pl,
        az=float(az),
        el=float(el),
        center=center,
        radius=r_used,
        ortho=True,
        margin=1.0,
        bounds=None,
    )

    if roll:
        try:
            pl.camera.Roll(float(roll))
        except Exception:
            pass

    try:
        pl.enable_parallel_projection()
    except Exception:
        pass

    right_v, up_v, fwd_v = _basis_from_camera(pl)

    pts = np.asarray(mesh_draw.points, dtype=float)

    r_min, r_max, u_min, u_max, h_span, v_span = _project_spans_exact(pts, center, right_v, up_v)

    aspect = (w_r / h_r) if h_r else 1.0

    proj_aspect = (h_span / v_span) if v_span > 1e-9 else 1.0

    if proj_aspect > aspect:
        ps_exact = h_span / (2.0 * aspect)
    else:
        ps_exact = v_span / 2.0

    ps_exact = max(ps_exact, 1e-6)

    if forced_parallel_scale is not None:
        ps = float(forced_parallel_scale)
    else:
        ps = ps_exact

    pl.camera.SetParallelScale(float(ps))

    try:
        pl.renderer.reset_camera_clipping_range()
    except Exception:
        pass

    try:
        pl.enable_eye_dome_lighting()
    except Exception:
        pass

    pl.render()

    parallel_scale = float(pl.camera.GetParallelScale())

    proj_points: dict[str, tuple[int, int]] = {}
    if points_world:
        try:
            ren = pl.renderer
            Wp, Hp = pl.ren_win.GetSize()
            for name, pt in points_world.items():
                pt = np.asarray(pt, float).ravel()
                if pt.size < 3:
                    continue
                ren.SetWorldPoint(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
                ren.WorldToDisplay()
                x, y, _ = ren.GetDisplayPoint()
                u_px = int(round(x))
                v_px = int(round(Hp - y))
                proj_points[name] = (u_px, v_px)
        except Exception:
            proj_points = {}

    img = pl.screenshot(return_img=True)
    if img.shape[2] == 4:
        img = img[..., :3]
    img = np.ascontiguousarray(img, dtype=np.uint8)

    meta = {
        "view_name": view_name,
        "az": float(az),
        "el": float(el),
        "roll": float(roll),
        "center": center.tolist(),
        "parallel_scale": parallel_scale,
        "render_window_size": (w_r, h_r),
        "right": right_v.tolist(),
        "up": up_v.tolist(),
        "fwd": fwd_v.tolist(),
        "proj_points": proj_points,
    }

    pl.close()
    return img, meta


def render_views_pixmaps_by_sizes(
        mesh: pv.PolyData,
        *,
        sizes_by_key: dict[str, tuple[int, int]],
        background: str = "white",
        color: str = "white",
        zoom_fact: float = 1.0,
        multi_samples: int = 0,
        supersample: float = 4.0,
        background_by_key=None,
        dist_scale: float = 1.9,
        meta_out: dict[str, dict] | None = None,
        views_spec: dict[str, Any] | None = None,
        points_world: dict[str, np.ndarray] | None = None,
) -> dict[str, QPixmap]:
    views = get_views_spec() if views_spec is None else views_spec
    if meta_out is not None:
        meta_out.clear()

    ss = max(1.0, float(supersample))
    pixmaps: dict[str, QPixmap] = {}

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.array([(xmax + xmin) / 2.0, (ymax + ymin) / 2.0, (zmax + zmin) / 2.0], float)

    pts = np.asarray(mesh.points, float)

    zoom = max(float(zoom_fact), 1e-6) if zoom_fact else 1.0

    view_data: list[dict] = []

    for key, v in views.items():
        if key not in sizes_by_key:
            continue

        if isinstance(v, dict):
            az = float(v.get("az", 0.0))
            el = float(v.get("el", 0.0))
            roll = float(v.get("roll", 0.0))
        else:
            az = float(v[0])
            el = float(v[1])
            roll = float(v[2]) if len(v) > 2 else 0.0

        w_t, h_t = sizes_by_key[key]
        w_t = max(1, int(w_t))
        h_t = max(1, int(h_t))

        w_r = max(1, int(np.ceil(w_t * ss)))
        h_r = max(1, int(np.ceil(h_t * ss)))

        right, up, _ = _basis_from_azel_roll(az, el, roll)
        r_min, r_max, u_min, u_max, h_span, v_span = _project_spans_exact(pts, center, right, up)

        aspect = w_r / h_r if h_r else 1.0
        proj_aspect = (h_span / v_span) if v_span > 1e-9 else 1.0

        if proj_aspect > aspect:
            world_per_px_req = h_span / w_r
        else:
            world_per_px_req = v_span / h_r

        world_per_px_req = world_per_px_req / zoom

        view_data.append({
            "key": key,
            "az": az,
            "el": el,
            "roll": roll,
            "w_t": w_t,
            "h_t": h_t,
            "w_r": w_r,
            "h_r": h_r,
            "world_per_px": world_per_px_req,
            "h_span": h_span,
            "v_span": v_span,
        })

    if not view_data:
        return pixmaps

    world_per_px_global = max(vd["world_per_px"] for vd in view_data) * GLOBAL_RULER_MARGIN

    for vd in view_data:
        key = vd["key"]
        az = vd["az"]
        el = vd["el"]
        roll = vd["roll"]
        w_t = vd["w_t"]
        h_t = vd["h_t"]
        w_r = vd["w_r"]
        h_r = vd["h_r"]

        forced_ps = world_per_px_global * h_r / 2.0

        bg = background
        if isinstance(background_by_key, dict):
            bg = background_by_key.get(key, background)
        elif callable(background_by_key):
            try:
                bg = background_by_key(key)
            except Exception:
                bg = background

        img, meta = _render_single_view_img_and_meta(
            mesh,
            view_name=key,
            forced_parallel_scale=float(forced_ps),
            window_size=(w_r, h_r),
            background=bg,
            color=color,
            az=az,
            el=el,
            roll=roll,
            zoom_fact=1.0,
            multi_samples=int(multi_samples),
            dist_scale=float(dist_scale),
            points_world=points_world,
        )

        pm_big = _np_to_qpixmap_rgb(img)
        pm = pm_big.scaled(w_t, h_t, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        pixmaps[key] = pm

        pp = meta.get("proj_points") or {}
        if isinstance(pp, dict) and pp:
            sx = w_t / max(1, int(w_r))
            sy = h_t / max(1, int(h_r))
            meta["proj_points"] = {
                name: (int(round(u * sx)), int(round(v * sy)))
                for name, (u, v) in pp.items()
            }

        meta["render_window_size"] = (w_r, h_r)
        meta["window_size"] = (w_t, h_t)
        meta["world_per_px"] = float(world_per_px_global)
        meta["background"] = bg

        if meta_out is not None:
            meta_out[key] = meta

    return pixmaps