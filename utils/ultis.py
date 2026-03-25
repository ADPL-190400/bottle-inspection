from PyQt6.QtGui import QImage, QPixmap
import cv2


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    
    # qt_image = qt_image.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
    return QPixmap.fromImage(qt_image)