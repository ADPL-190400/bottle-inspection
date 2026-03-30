from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal, QDir
from PyQt6.QtGui import QFileSystemModel
import queue
from core.path_manager import BASE_DIR
import os
import re
import threading
import cv2
from hardware.camera.batch_camera import BatchCamera
from training.patchcore_memory_bank import build_bank


# =========================================================================== #
#  BUILD WORKER (chạy build_bank trên QThread riêng)                          #
# =========================================================================== #
class BuildBankWorker(QThread):
    finished = pyqtSignal(bool, str)   # (success, message)

    def run(self):
        try:
            build_bank()
            self.finished.emit(True, "✅ Build Memory Bank hoàn tất.")
        except Exception as e:
            self.finished.emit(False, f"❌ Build thất bại: {e}")


# =========================================================================== #
#  GET DATA TAB                                                                #
# =========================================================================== #
class GetDataTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", "ui", "get_data_tab.ui")
        uic.loadUi(ui_path, self)

        self.btn_start_get_data.clicked.connect(self.start_get_data)
        self.btn_stop_get_data.clicked.connect(self.stop_get_data)
        self.build_memory_bank.clicked.connect(self.build_memory)

        self.data_manager   = None
        self.is_initialized = False
        self.is_saving      = False
        self._build_worker  = None

        # QFileSystemModel cho treeView_img
        self._fs_model = QFileSystemModel()
        self._fs_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
        self.treeView_img.setModel(self._fs_model)
        # Chỉ hiện cột Name, ẩn Size / Type / Date
        self.treeView_img.setColumnHidden(1, True)
        self.treeView_img.setColumnHidden(2, True)
        self.treeView_img.setColumnHidden(3, True)
        self.treeView_img.setHeaderHidden(False)

    # ----------------------------------------------------------------------- #
    def _set_tree_root(self, path: str):
        """Trỏ treeView_img vào thư mục đã chọn."""
        root_index = self._fs_model.setRootPath(path)
        self.treeView_img.setRootIndex(root_index)
        self.treeView_img.expandAll()

    # ----------------------------------------------------------------------- #
    def _set_data_buttons_enabled(self, enabled: bool):
        """Enable/disable các nút liên quan đến thu thập dữ liệu."""
        self.btn_start_get_data.setEnabled(enabled)
        self.btn_stop_get_data.setEnabled(enabled)
        self.build_memory_bank.setEnabled(enabled)

    # ----------------------------------------------------------------------- #
    def init_camera(self):
        if self.is_initialized:
            print("[GetData] Camera đã khởi tạo rồi.")
            return
        try:
            self.data_manager = GetDataManager()
            self.data_manager.start()
            self.is_initialized = True
            print("[GetData] ✅ Camera khởi tạo thành công.")
        except Exception as e:
            print(f"[GetData] ❌ Khởi tạo thất bại: {e}")

    # ----------------------------------------------------------------------- #
    def start_get_data(self):
        if self._build_worker and self._build_worker.isRunning():
            print("[GetData] Đang build Memory Bank, không thể thu thập.")
            return

        self.init_camera()
        if not self.is_initialized:
            print("[GetData] Chưa init camera.")
            return
        if self.is_saving:
            print("[GetData] Đang thu thập rồi.")
            return

        save_dir = self._get_save_dir()
        if not save_dir:
            return

        self.data_manager.set_save_dir(save_dir)
        self.data_manager.is_saving = True
        self.is_saving = True
        print(f"[GetData] ▶ Bắt đầu thu thập → {save_dir}")

    # ----------------------------------------------------------------------- #
    def stop_get_data(self):
        if not self.is_saving:
            return
        self.data_manager.is_saving = False
        self.is_saving = False
        print("[GetData] ⏹ Dừng thu thập.")

    # ----------------------------------------------------------------------- #
    def _get_save_dir(self) -> str | None:
        line_edit = getattr(self, "save_dir_input", None)
        if line_edit:
            path = line_edit.text().strip()
            if path:
                os.makedirs(path, exist_ok=True)
                self._set_tree_root(path)
                return path

        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh")
        if not path:
            print("[GetData] Chưa chọn thư mục.")
            return None

        self._set_tree_root(path)
        return path

    # ----------------------------------------------------------------------- #
    def build_memory(self):
        if self._build_worker and self._build_worker.isRunning():
            print("[GetData] Đang build rồi.")
            return

        self.stop_get_data()
        self._set_data_buttons_enabled(False)
        print("[GetData] 🔨 Bắt đầu build Memory Bank...")

        self._build_worker = BuildBankWorker()
        self._build_worker.finished.connect(self._on_build_finished)
        self._build_worker.start()

    # ----------------------------------------------------------------------- #
    def _on_build_finished(self, success: bool, message: str):
        print(f"[GetData] {message}")
        self._set_data_buttons_enabled(True)

    # ----------------------------------------------------------------------- #
    def closeEvent(self, event):
        if self._build_worker and self._build_worker.isRunning():
            self._build_worker.quit()
            self._build_worker.wait(3000)
        if self.data_manager:
            self.data_manager.stop()
            self.data_manager.join(timeout=3)
        super().closeEvent(event)


# =========================================================================== #
#  GET DATA MANAGER                                                            #
# =========================================================================== #
class GetDataManager(threading.Thread):
    """
    Chạy BatchCamera, khi có trigger và is_saving=True thì lưu ảnh.

    Cấu trúc thư mục (phẳng, không tạo subfolder):
        save_dir/
            cam1_0001.jpg  cam2_0001.jpg  ...
            cam1_0002.jpg  cam2_0002.jpg  ...
            ...

    Khi start lại cùng thư mục, tự động đếm file hiện có
    và ghi tiếp từ index tiếp theo — không ghi đè.
    """

    def __init__(self):
        super().__init__(daemon=True, name="GetDataManager")
        self.frames_queue   = queue.Queue(maxsize=1)
        self._stop_event    = threading.Event()
        self.is_saving      = False
        self.save_dir       = None
        self._trigger_count = 0

    # ----------------------------------------------------------------------- #
    def set_save_dir(self, path: str):
        """Đặt thư mục lưu và tự resume index từ file đã có."""
        self.save_dir = path
        self._trigger_count = self._scan_last_index(path)
        if self._trigger_count > 0:
            print(f"[GetDataManager] 🔄 Resume từ trigger_{self._trigger_count:04d} "
                  f"(đã có {self._trigger_count} trigger)")
        else:
            print("[GetDataManager] 🆕 Thư mục trống, bắt đầu từ 0001")

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _scan_last_index(directory: str) -> int:
        """
        Quét thư mục, tìm số trigger lớn nhất từ các file có dạng
        cam{N}_{index}.jpg  →  trả về index lớn nhất (hoặc 0 nếu trống).
        """
        if not os.path.isdir(directory):
            return 0
        pattern = re.compile(r"^cam\d+_(\d+)\.jpg$", re.IGNORECASE)
        max_index = 0
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                idx = int(m.group(1))
                if idx > max_index:
                    max_index = idx
        return max_index

    # ----------------------------------------------------------------------- #
    def stop(self):
        self._stop_event.set()

    # ----------------------------------------------------------------------- #
    def run(self):
        thread_camera = None
        try:
            try:
                thread_camera = BatchCamera(self.frames_queue, self._stop_event)
                thread_camera.start()
                print("[GetDataManager] Camera started.")
            except Exception as e:
                print(f"[GetDataManager] ❌ Camera lỗi: {e}")
                return

            while not self._stop_event.is_set():
                try:
                    trigger_id, frames = self.frames_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if self.is_saving and self.save_dir:
                    self._save_trigger(frames)

        finally:
            if thread_camera:
                thread_camera.join(timeout=3)
            print("[GetDataManager] Stopped.")

    # ----------------------------------------------------------------------- #
    def _save_trigger(self, frames: list):
        self._trigger_count += 1

        saved = 0
        for cam_id, frame in enumerate(frames):
            if frame is None:
                continue
            filename = f"cam{cam_id + 1}_{self._trigger_count:04d}.jpg"
            path = os.path.join(self.save_dir, filename)
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved += 1

        print(f"[GetDataManager] 💾 trigger_{self._trigger_count:04d} → {saved} ảnh")