# src/artifact_app/viewer/views_layout.py
from __future__ import annotations

def get_views_positions() -> dict[str, tuple[int, int]]:
    """
    פריסה עקבית (כמו ב-MainWindow):
    TL בפינה שמאל-עליון; שורת האמצע: ML, MC, MR; BR בפינה ימין-תחתון.
    """
    return {
        "TL": (0, 0),
        "ML": (1, 0),
        "MC": (1, 1),
        "MR": (1, 2),
        "BR": (2, 2),
    }
