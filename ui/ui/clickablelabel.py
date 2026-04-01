from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, pyqtSignal, Qt 
from PyQt6.QtGui import QMouseEvent

class ClickableLabel(QLabel):
    clicked = pyqtSignal(QPoint)  # Tín hiệu phát ra khi nhấn chuột

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)