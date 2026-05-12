# src/artifact_app/viewer/views_spec.py
from __future__ import annotations
from .main_view import set_view_azel

# ==============================================================================
# המרת זוויות המבט מ-MATLAB (az, el) למערכת הצירים של PyVista
# ==============================================================================

VIEWS = {
    # Top Left (מבט מלמעלה) - דורש roll מתאים כדי לשמור על כיווני הצירים כמו ב-MATLAB
    "TL": dict(az=0.0, el=90, roll=0),

    # Mid Left (צד 1)
    "ML": dict(az=90.0, el=0.0, roll=0.0),

    # Mid Center (חזית / מבט ראשי)
    "MC": dict(az=0.0, el=0.0, roll=0.0),

    # Mid Right (צד 2)
    "MR": dict(az=180, el=0.0, roll=0.0),

    # Bottom Right (מבט מלמטה)
    "BR": dict(az=0.0, el=-90, roll=0),

    # ---> התיקון: הוספנו ישירות את המבט האחורי עבור חלון ה-Live Views <---
    "MC_BACK": dict(az=180.0, el=0.0, roll=0.0),
}

# שמות נרדפים (Aliases) ותאימות לאחור
ALIASES = {
    "center": "MC",
    "main": "MC",
    "front": "MC",
    "left": "ML",
    "right": "MR",
    "top": "TL",
    "bottom": "BR",
    "mc_back": "MR",
    "back": "MR",

    "c": "MC", "l": "ML", "r": "MR", "t": "TL", "b": "BR",
}


def get_views_spec() -> dict[str, dict[str, float]]:
    return VIEWS.copy()


def get_aliases() -> dict[str, str]:
    return ALIASES.copy()


def look_named(plotter, name: str, center, radius, *, ortho: bool = True,
               margin: float = 1.06, bounds=None) -> None:
    """מציב את המצלמה לפי שם המבט (כולל תמיכה ב-ALIASES)."""
    key = name if name in VIEWS else ALIASES.get(name.lower())
    if key not in VIEWS:
        raise KeyError(f"Unknown view name: {name!r}")

    spec = VIEWS[key]
    roll = spec.get("roll", 0.0)

    set_view_azel(
        plotter,
        float(spec["az"]), float(spec["el"]),
        center=center,
        radius=radius,
        ortho=ortho,
        bounds=bounds,
        margin=margin,
    )

    if roll != 0:
        plotter.camera.Roll(roll)