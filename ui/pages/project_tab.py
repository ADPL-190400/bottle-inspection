from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import queue
import numpy as np
from core.path_manager import BASE_DIR
import os
import threading
import time
import cv2
from utils.ultis import convert_cv_qt




BATCH_COLLECTION_TIMEOUT = 1/30

class ProjectTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", 'ui', "project_tab.ui")
        uic.loadUi(ui_path, self)

        