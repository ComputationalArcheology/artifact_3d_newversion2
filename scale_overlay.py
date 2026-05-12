# scale_overlay.py
from __future__ import annotations
import math

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QFont


def _pick_nice_length(target_mm: float) -> float:
    """
    בוחר אורך 'יפה' בסגנון 1/2/5×10^n, קרוב ל-target_mm.
    """
    if target_mm <= 0:
        return 1.0

    exp = math.floor(math.log10(target_mm))
    base = 10 ** exp
    candidates = [1 * base, 2 * base, 5 * base, 10 * base]

    # הכי קרוב ליעד
    return min(candidates, key=lambda x: abs(x - target_mm))


def draw_scale_bar_on_pixmap(
    pm: QPixmap,
    meta: dict,
    *,
    frac_of_width: float = 0.28,   # בערך רבע רוחב, כמו במטלב
    where: str = "bottom",
) -> QPixmap:
    """
    מצייר סרגל קנה מידה על QPixmap לפי meta של המבט (parallel_scale + window_size).

    הערה: זה נכון *רק* למצלמה אורתוגרפית (ParallelProjection).
    """
    if pm.isNull():
        return pm
    if not isinstance(meta, dict) or "parallel_scale" not in meta or "window_size" not in meta:
        return pm

    ps = float(meta["parallel_scale"])
    w, h = map(int, meta["window_size"])
    if w <= 0 or h <= 0 or ps <= 0:
        return pm

    aspect = w / h
    world_w = 2.0 * ps * aspect          # רוחב העולם שנראה במבט
    mm_per_px = world_w / w              # כמה מ״מ לכל פיקסל (X)

    # אורך סרגל "יעד" במ״מ
    target_mm = max(1e-6, world_w * float(frac_of_width))
    scal_mm = _pick_nice_length(target_mm)

    scal_px = int(round(scal_mm / max(mm_per_px, 1e-12)))
    scal_px = max(40, min(scal_px, int(w * 0.85)))

    # מספר מקטעים כמו MATLAB-ish
    if scal_px >= 160:
        sl = 5
    elif scal_px >= 110:
        sl = 3
    else:
        sl = 2

    seg_px = max(1, scal_px // sl)
    bar_w = seg_px * sl
    bar_h = max(6, int(h * 0.035))

    margin = max(8, int(min(w, h) * 0.04))
    x0 = (w - bar_w) // 2

    if where == "bottom":
        y0 = h - margin - bar_h
        text_y = y0 - int(bar_h * 1.2)
    else:
        y0 = margin
        text_y = y0 + bar_h + int(bar_h * 0.6)

    # טקסט כמו MATLAB: <10mm => mm, אחרת cm
    if scal_mm < 10:
        label = f"{int(round(scal_mm))} mm"
    else:
        label = f"{int(round(scal_mm / 10))} cm"

    out = pm.copy()
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # מסגרת עדינה
    painter.setPen(QPen(Qt.black, 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(QRectF(x0 - 1, y0 - 1, bar_w + 2, bar_h + 2))

    # מקטעים שחור/לבן
    painter.setPen(Qt.NoPen)
    for i in range(sl):
        xi = x0 + i * seg_px
        painter.setBrush(Qt.black if (i % 2 == 0) else Qt.white)
        painter.drawRect(QRectF(xi, y0, seg_px, bar_h))

    # טקסט
    painter.setPen(QPen(Qt.black, 1))
    # טקסט
    painter.setPen(QPen(Qt.black, 1))

    # ✅ גופן קטן יותר (רק הטיפוגרפיה משתנה, לא הסרגל)
    # 0.03..0.035 עובד טוב לרוב; התחל מ-0.032
    font_px = max(9, int(h * 0.032))
    font = QFont("Andalus")
    font.setPixelSize(font_px)  # חשוב: PixelSize יציב יותר מ-pointSize
    font.setBold(True)
    painter.setFont(font)

    # תיבת טקסט קומפקטית סביב ה-y (לא 60px קבוע)
    box_h = int(font_px * 2.2)
    painter.drawText(
        QRectF(0, max(0, text_y - box_h / 2), w, box_h),
        Qt.AlignHCenter | Qt.AlignVCenter,
        label,
    )

    return out
