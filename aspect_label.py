# artifact_app/gui/aspect_label.py
"""
רכיבי תצוגת תמונה (QLabel) מותאמים.
ה-Layout מחשב את יחס הגובה-רוחב (בדומה להתנהגות ב-MATLAB), לכן אין צורך בשמירה כפולה על היחס וניתן למנוע מרווחים מיותרים.
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class AspectPixmapLabel(QLabel):
    """
    מציג תמונה ומותח אותה למלא את שטח התא ללא שמירה על יחס גובה-רוחב.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self._keep_aspect = False

    def setPixmap(self, pix: QPixmap) -> None:
        self._pix = pix
        super().setPixmap(self._scaled_pix())

    def resizeEvent(self, e):
        if self._pix:
            super().setPixmap(self._scaled_pix())
        return super().resizeEvent(e)

    def _scaled_pix(self) -> QPixmap:
        if self._pix is None:
            return QPixmap()

        if self._keep_aspect:
            return self._pix.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        else:
            # ה-Layout כבר מחשב את הגודל, לכן מותחים למלא את השטח
            return self._pix.scaled(
                self.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )

    def setKeepAspectRatio(self, keep: bool) -> None:
        """
        משנה את התנהגות המתיחה:
        True - שומר יחס (עלול ליצור מרווחים), False - מותח למלא את התא.
        """
        self._keep_aspect = bool(keep)
        if self._pix:
            super().setPixmap(self._scaled_pix())


class ExactPixmapLabel(QLabel):
    """
    מציג תמונה בגודלה המקורי ללא scaling.
    מתאים כשהתמונה כבר מרונדרת מראש בגודל התא המדויק.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pix: QPixmap) -> None:
        self._pix = pix
        super().setPixmap(pix)

    def hasPixmap(self) -> bool:
        return self._pix is not None

    def originalPixmap(self) -> QPixmap | None:
        return self._pix