# src/artifact_app/viewer/scale_bar.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QFont


# ------------------------------------------------------------
# שעתוק הלוגיקה המדויקת מ-MATLAB לבחירת גודל סרגל קנה המידה
# ------------------------------------------------------------
def _scale_size_for_gui_matlab(s0: float) -> tuple[int, float]:
    """
    מחזיר:
      sl      = מספר מקטעים מומלץ (2/3/5)
      scal_mm = אורך כולל "יפה" ועגול במ"מ
    """
    sc = 1.0
    not_done = True
    s1 = 3
    units = 1.0

    s0 = float(s0)
    if not math.isfinite(s0) or s0 <= 0:
        return 3, 3.0

    while not_done:
        x = sc * s0
        if 0 < x < 3:
            sc *= 10.0
        elif 3 <= x < 5:
            not_done = False
            s1, units = 3, 1.0
        elif 5 <= x < 10:
            not_done = False
            s1, units = 5, 1.0
        elif 10 <= x < 15:
            not_done = False
            s1, units = 5, 2.0
        elif 15 <= x < 20:
            not_done = False
            s1, units = 3, 5.0
        elif 20 <= x < 30:
            not_done = False
            s1, units = 2, 10.0
        elif x >= 30:
            sc *= 0.1

    scal_mm = (s1 * units) / sc
    return int(s1), float(scal_mm)


def _world_width_visible_mm(meta: dict) -> float:
    """
    מחשב את רוחב העולם הנראה במבט אורתוגרפי מתוך המטא-דאטה של המצלמה.
    """
    ps = float(meta.get("parallel_scale", 0.0))
    w, h = map(int, meta.get("window_size", (0, 0)))
    if ps <= 0 or w <= 0 or h <= 0:
        return 0.0
    aspect = w / h
    return 2.0 * ps * aspect


def make_scale_pixmap(
        w_scale: int,
        h_scale: int,
        ref_meta: dict,
        *,
        target_frac: float = 0.80,
        dpr: float = 1.0,
        s0_mm: float | None = None,
        debug: bool = False,
        bg: str | Qt.GlobalColor = Qt.white,
        font_size: int | None = None,
) -> QPixmap:
    w_scale = max(1, int(w_scale))
    h_scale = max(1, int(h_scale))

    dpr = float(dpr) if dpr else 1.0
    pix = QPixmap(int(round(w_scale * dpr)), int(round(h_scale * dpr)))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.white)
    pix.fill(bg)

    world_w = _world_width_visible_mm(ref_meta)
    if world_w <= 1e-9:
        return pix

    w_ref, h_ref = map(int, ref_meta.get("window_size", (0, 0)))
    if w_ref <= 0 or h_ref <= 0:
        return pix

    mm_per_px_ref = world_w / float(w_ref)

    if s0_mm is not None and math.isfinite(float(s0_mm)) and float(s0_mm) > 0:
        s0 = max(1.0, math.floor(float(s0_mm)))
    else:
        s0 = max(1.0, math.floor(world_w))

    out = _scale_size_for_gui_matlab(s0)
    if len(out) == 3:
        sl, scal_mm, _sc = out
    else:
        sl, scal_mm = out

    bar_px_ref = scal_mm / mm_per_px_ref
    max_bar_px = max(1.0, float(target_frac) * w_scale)

    if debug:
        print("[SCALE make_scale_pixmap]",
              "world_w_mm=", world_w,
              "s0_mm_in=", s0_mm,
              "s0_used=", s0,
              "cell_w,h=", (w_scale, h_scale),
              "ref_w,h=", (w_ref, h_ref),
              "ps=", ref_meta.get("parallel_scale"),
              "mm_per_px_ref=", mm_per_px_ref,
              "sl=", sl, "scal_mm=", scal_mm,
              "bar_px_ref=", bar_px_ref,
              "max_bar_px=", max_bar_px)

    guard = 0
    while bar_px_ref > max_bar_px and s0 > 1 and guard < 400:
        s0 -= 1.0
        out = _scale_size_for_gui_matlab(s0)
        if len(out) == 3:
            sl, scal_mm, _sc = out
        else:
            sl, scal_mm = out
        bar_px_ref = scal_mm / mm_per_px_ref
        guard += 1

    bar_px = min(bar_px_ref, max_bar_px)

    frac = bar_px / float(w_scale)
    if (frac <= 0.10 and sl == 5) or (frac <= 0.05 and sl == 3):
        sl = 2

    seg_px = bar_px / float(sl)

    x0 = (w_scale - bar_px) * 0.5
    bar_h = int(max(6, min(h_scale * 0.30, seg_px * 0.30)))
    y0 = int(h_scale * 0.55)

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)

    for i in range(sl):
        rect = QRectF(x0 + i * seg_px, y0, seg_px, bar_h)
        p.fillRect(rect, QBrush(Qt.white if (i % 2 == 0) else Qt.black))
        p.setPen(QPen(Qt.black, 1))
        p.drawRect(rect)

    if scal_mm < 10:
        txt = f"{int(round(scal_mm))} mm"
    else:
        txt = f"{scal_mm / 10:g} cm"

    if font_size is not None:
        fs = max(6, int(font_size))
    else:
        # הגבלת גודל הפונט של סרגל קנה המידה כדי למנוע עיוותים
        calculated_size = int(h_scale * 0.15)
        fs = max(11, min(16, calculated_size))

    font = QFont("Andalus", fs)
    p.setFont(font)
    p.setPen(QPen(Qt.black, 1))
    p.drawText(QRectF(0, 0, w_scale, max(1, y0 - 2)),
               Qt.AlignHCenter | Qt.AlignVCenter, txt)

    p.end()
    return pix