from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import queue
import numpy as np
from core.path_manager import BASE_DIR
import os
<<<<<<< HEAD
import threading
=======
>>>>>>> 5fe2763 (update 2703)
import time
import cv2
from utils.ultis import convert_cv_qt
from core.pipeline_maneger import PipelineManager


<<<<<<< HEAD

BATCH_COLLECTION_TIMEOUT = 1/30
=======
BATCH_COLLECTION_TIMEOUT = 1 / 30

>>>>>>> 5fe2763 (update 2703)

class ProcessTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        ui_path = os.path.join(BASE_DIR, "ui", 'ui', "process_tab.ui")
=======
        ui_path = os.path.join(BASE_DIR, "ui", "ui", "process_tab.ui")
>>>>>>> 5fe2763 (update 2703)
        uic.loadUi(ui_path, self)

        self.btn_play.clicked.connect(self.load_project_yaml)
        self.btn_stop.clicked.connect(self.stop_all_threads)

<<<<<<< HEAD
        self.is_playing = False
        self.pipeline_manager = None
        self.frame_updater = None
        self.show_queue = {str(cam_id): queue.Queue(maxsize=1) for cam_id in range(1,6)}

=======
        self.is_playing       = False
        self.pipeline_manager = None
        self.frame_updater    = None

        # show_queue nhận tuple (frame_bgr, is_ok)
        #   is_ok = None  → raw frame chưa có kết quả AI
        #   is_ok = True  → cam này OK
        #   is_ok = False → cam này NG
        self.show_queue = {
            str(cam_id): queue.Queue(maxsize=1) for cam_id in range(1, 6)
        }

    # ----------------------------------------------------------------------- #
>>>>>>> 5fe2763 (update 2703)
    def load_project_yaml(self):
        if self.is_playing:
            print("Pipeline already running")
            return
        print("Load project yaml")
<<<<<<< HEAD

        self.is_playing = True

        # Start frame updater thread 
        self.frame_updater = UpdateFrameThread(self.show_queue)
        self.frame_updater.frame_ready.connect(self.update_frame)
        self.frame_updater.start()

        # Start pipeline manager
        self.pipeline_manager = PipelineManager(self.show_queue)
        self.pipeline_manager.start()

=======
        self.is_playing = True

        self.frame_updater = UpdateFrameThread(self.show_queue)
        self.frame_updater.frame_ready.connect(self.update_frame)
        self.frame_updater.result_batch.connect(self.update_classification)
        self.frame_updater.start()

        self.pipeline_manager = PipelineManager(self.show_queue)
        self.pipeline_manager.start()

    # ----------------------------------------------------------------------- #
>>>>>>> 5fe2763 (update 2703)
    def stop_all_threads(self):
        if not self.is_playing:
            return
        try:
            print("Stopping all threads...")

<<<<<<< HEAD
            # Stop pipeline manager
=======
>>>>>>> 5fe2763 (update 2703)
            if self.pipeline_manager:
                self.pipeline_manager.stop()
                self.pipeline_manager.join()
                self.pipeline_manager = None

<<<<<<< HEAD
            # Stop frame updater
=======
>>>>>>> 5fe2763 (update 2703)
            if self.frame_updater:
                self.frame_updater.stop()
                self.frame_updater.quit()
                self.frame_updater.wait()
                self.frame_updater = None

<<<<<<< HEAD
            # Clear queues
=======
>>>>>>> 5fe2763 (update 2703)
            for q in self.show_queue.values():
                with q.mutex:
                    q.queue.clear()

            self.is_playing = False
            print("All threads stopped")

        except Exception as e:
            print("Error stopping threads:", e)

<<<<<<< HEAD



    
    def update_frame(self, cam_id, frame):
        
        """Cập nhật frame lên QLabel"""
=======
    # ----------------------------------------------------------------------- #
    def update_frame(self, cam_id: str, frame: np.ndarray):
        """Cập nhật frame lên QLabel."""
>>>>>>> 5fe2763 (update 2703)
        cam_widget = getattr(self, f"cam_{cam_id}", None)
        if cam_widget is None:
            return
        target_size = cam_widget.size()
<<<<<<< HEAD
        
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






=======
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        qt_img = convert_cv_qt(frame)
        scaled = qt_img.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        cam_widget.setPixmap(scaled)

    # ----------------------------------------------------------------------- #
    def update_classification(self, is_ok: bool):
        """Cập nhật kết quả phân loại batch lên UI."""
        label = "OK" if is_ok else "NG"
        color = "green" if is_ok else "red"
        self.text_result.setText(label)
        self.text_result.setStyleSheet(f"color: {color}; font-weight: bold;")


# =========================================================================== #
#  UPDATE FRAME THREAD                                                         #
# =========================================================================== #
class UpdateFrameThread(QThread):
    frame_ready  = pyqtSignal(str, np.ndarray)   # (cam_id, frame_bgr)
    result_batch = pyqtSignal(bool)              # True=OK, False=NG

    def __init__(self, show_queue: dict):
        super().__init__()
        self.show_queue = show_queue
        self._running   = True

        # Giữ is_ok mới nhất của từng cam (None = chưa có kết quả AI)
        self._last_is_ok = {cam_id: None for cam_id in show_queue}

    def run(self):
        while self._running:
            got_ai_result = False   # có cam nào nhận được kết quả AI trong vòng này không

            for cam_id, q in self.show_queue.items():
                try:
                    # ✅ Unpack đúng tuple (frame_bgr, is_ok) từ pipeline
                    frame, is_ok = q.get_nowait()
                except queue.Empty:
                    continue
                except (TypeError, ValueError):
                    # Phòng trường hợp queue cũ chỉ có frame (không có is_ok)
                    continue

                # Emit frame lên UI
                self.frame_ready.emit(cam_id, frame)

                # Cập nhật is_ok mới nhất của cam này
                if is_ok is not None:
                    self._last_is_ok[cam_id] = is_ok
                    got_ai_result = True

            # Chỉ emit result_batch khi có kết quả AI mới
            # và tất cả cam đã có kết quả (không còn None)
            if got_ai_result:
                known = [v for v in self._last_is_ok.values() if v is not None]
                if known:
                    batch_ok = all(known)
                    self.result_batch.emit(batch_ok)

            time.sleep(BATCH_COLLECTION_TIMEOUT / 2)

    def stop(self):
        self._running = False
>>>>>>> 5fe2763 (update 2703)
