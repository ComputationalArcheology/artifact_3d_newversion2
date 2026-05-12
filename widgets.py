# src/artifact_app/gui/widgets.py
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QToolButton


class ArtifactToolButton(QToolButton):
    """
    כפתור תפריט נפתח (QToolButton)
    העיצוב נשלט כעת מקובץ ה-QSS הגלובלי (style.qss)
    """
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        if text:
            self.setText(text)

        self.setCursor(Qt.PointingHandCursor)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMinimumHeight(30)


class ArtifactButton(QPushButton):
    """
    כפתור "ברירת מחדל" ל-Artifact
    העיצוב נשלט כעת מקובץ ה-QSS הגלובלי (style.qss)
    """
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)

        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(30)