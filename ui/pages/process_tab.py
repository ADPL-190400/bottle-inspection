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
from core.pipeline_maneger import PipelineManager



BATCH_COLLECTION_TIMEOUT = 1/30

class ProcessTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", 'ui', "process_tab.ui")
        uic.loadUi(ui_path, self)

        self.btn_play.clicked.connect(self.load_project_yaml)
        self.btn_stop.clicked.connect(self.stop_all_threads)

        self.is_playing = False
        self.pipeline_manager = None
        self.frame_updater = None
        self.show_queue = {str(cam_id): queue.Queue(maxsize=1) for cam_id in range(1,6)}

    def load_project_yaml(self):
        if self.is_playing:
            print("Pipeline already running")
            return
        print("Load project yaml")

        self.is_playing = True

        # Start frame updater thread 
        self.frame_updater = UpdateFrameThread(self.show_queue)
        self.frame_updater.frame_ready.connect(self.update_frame)
        self.frame_updater.start()

        # Start pipeline manager
        self.pipeline_manager = PipelineManager(self.show_queue)
        self.pipeline_manager.start()

    def stop_all_threads(self):
        if not self.is_playing:
            return
        try:
            print("Stopping all threads...")

            # Stop pipeline manager
            if self.pipeline_manager:
                self.pipeline_manager.stop()
                self.pipeline_manager.join()
                self.pipeline_manager = None

            # Stop frame updater
            if self.frame_updater:
                self.frame_updater.stop()
                self.frame_updater.quit()
                self.frame_updater.wait()
                self.frame_updater = None

            # Clear queues
            for q in self.show_queue.values():
                with q.mutex:
                    q.queue.clear()

            self.is_playing = False
            print("All threads stopped")

        except Exception as e:
            print("Error stopping threads:", e)




    
    def update_frame(self, cam_id, frame):
        
        """Cập nhật frame lên QLabel"""
        cam_widget = getattr(self, f"cam_{cam_id}", None)
        if cam_widget is None:
            return
        target_size = cam_widget.size()
        
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        
        qt_img = convert_cv_qt(frame)

        scaled = qt_img.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,  # giữ tỷ lệ
            Qt.TransformationMode.SmoothTransformation
        )

        
        cam_widget.setPixmap(scaled)






    
# ---------------- Update Frame ----------------
class UpdateFrameThread(QThread):
    frame_ready = pyqtSignal(str, np.ndarray)

    def __init__(self, show_queue):
        super().__init__()
        self.show_queue = show_queue
        self._running = True

    def run(self):
        
        while self._running:
        
            for cam_id, q in self.show_queue.items():
                    try:
                        frame = q.get_nowait()
                        # print(f"Got frame for cam {cam_id}")
                        self.frame_ready.emit(cam_id, frame)
                       
                    except queue.Empty:
                        pass

            time.sleep(BATCH_COLLECTION_TIMEOUT / 2)


    def stop(self):
        self._running = False






