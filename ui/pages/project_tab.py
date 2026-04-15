# coding=utf-8
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QStringListModel
from PyQt6.QtWidgets import QMessageBox
import queue
import numpy as np
import threading
from core.path_manager import BASE_DIR
import os
import json
import platform
from hardware.camera import mvsdk
from utils.ultis import convert_cv_qt
from hardware.gpio.trigger_input_camera import TriggerCamera
import time


BATCH_COLLECTION_TIMEOUT = 1 / 30


# ── Worker thread: lắng nghe trigger_queue và emit signal về UI thread ──
class TriggerWorker(QThread):
    triggered = pyqtSignal()          # signal không có tham số, gọi về main thread

    def __init__(self, trigger_queue: queue.Queue, stop_event: threading.Event, parent=None):
        super().__init__(parent)
        self.trigger_queue = trigger_queue
        self.stop_event    = stop_event

    def run(self):
        print("[TriggerWorker] Running... waiting for trigger")
        while not self.stop_event.is_set():
            try:
                if self.trigger_queue.empty():
                    time.sleep(0.01)
                    continue
                _ = self.trigger_queue.get_nowait()
                self.triggered.emit()          # ← an toàn: Qt tự marshal sang UI thread
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[TriggerWorker] {e}")


class ProjectTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", "ui", "project_tab.ui")
        uic.loadUi(ui_path, self)

        # Camera state
        self._dev_list         = []
        self._current_dev_info = None
        self._hCamera          = None
        self._pFrameBuffer     = None
        self._mono_camera      = False

        # Thread state
        self.stop_event        = threading.Event()
        self.trigger_queue     = queue.Queue(maxsize=1)
        self.thread_trigger_camera = None   # TriggerCamera (hardware GPIO)
        self._trigger_worker       = None   # TriggerWorker (queue listener)

        self._setup_initial_state()
        self._setup_project_list()
        self._connect_signals()
        self._list_camera()
        self._init_trigger_camera()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _setup_initial_state(self):
        self.text_name_project.setEnabled(False)
        self.btn_create_project.setEnabled(False)
        self.btn_execute_trigger.setEnabled(False)

    def _init_trigger_camera(self):
        """Khởi động TriggerCamera (GPIO) và TriggerWorker (queue → signal)."""
        # 1. Hardware trigger thread
        self.thread_trigger_camera = TriggerCamera(
            self.trigger_queue, stop_event=self.stop_event, offset=3
        )
        self.thread_trigger_camera.start()

        # 2. Worker lắng nghe queue, emit signal về UI thread
        self._trigger_worker = TriggerWorker(self.trigger_queue, self.stop_event, parent=self)
        self._trigger_worker.triggered.connect(self._on_execute_trigger_clicked)   # ← kết nối
        self._trigger_worker.start()

    def _connect_signals(self):
        self.btn_add_project.clicked.connect(self._on_add_project_clicked)
        self.btn_create_project.clicked.connect(self._on_create_project_clicked)
        self.btn_delete_project.clicked.connect(self._on_delete_project_clicked)
        self.devices_online.currentIndexChanged.connect(self._on_device_selected)
        self.btn_execute_trigger.clicked.connect(self._on_execute_trigger_clicked)
        self.btn_save_camera_config.clicked.connect(self._on_save_camera_config_clicked)
        self.list_project.selectionModel().selectionChanged.connect(self._on_project_selected)
        

        self.width_img.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.width_img,  step=16, minimum=16,  maximum=2448))
        self.height_img.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.height_img, step=4,  minimum=4,   maximum=2048))
        self.offset_x.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.offset_x,   step=2,  minimum=0,   maximum=2448))
        self.offset_y.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.offset_y,   step=2,  minimum=0,   maximum=2048))

    # ------------------------------------------------------------------
    # Camera enumeration
    # ------------------------------------------------------------------

    def _list_camera(self):
        self._dev_list = mvsdk.CameraEnumerateDevice()
        self.devices_online.clear()

        if not self._dev_list:
            print("No camera found")
            self.devices_online.addItem("-- No camera found --")
            self.btn_execute_trigger.setEnabled(False)
            return

        for dev_info in self._dev_list:
            link_name = dev_info.acSn.decode() if isinstance(dev_info.acSn, bytes) else dev_info.acSn
            self.devices_online.addItem(link_name)

        self.devices_online.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    def _on_device_selected(self, index: int):
        self._close_camera()

        if index < 0 or index >= len(self._dev_list):
            self._current_dev_info = None
            self.btn_execute_trigger.setEnabled(False)
            return

        self._current_dev_info = self._dev_list[index]
        print(f"Selected camera: {self._current_dev_info.acSn}")

        success = self._open_camera(self._current_dev_info)
        self.btn_execute_trigger.setEnabled(success)

    # ------------------------------------------------------------------
    # Camera open / close
    # ------------------------------------------------------------------

    def _open_camera(self, dev_info) -> bool:
        try:
            self._hCamera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as e:
            QMessageBox.critical(self, "Lỗi camera", f"CameraInit Failed:\n{e.message}")
            self._hCamera = None
            return False
        

        project_name = self.text_name_project.text().strip()


        if self._hCamera is None or self._current_dev_info is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có camera nào được mở!")
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir   = os.path.join(projects_root, project_name)

        link_name = self._current_dev_info.acSn
        if isinstance(link_name, bytes):
            link_name = link_name.decode()

        config_filename = f"{link_name}.config"
        path_camera = os.path.join(project_dir, "camera_config", config_filename)
        path_camera_default = os.path.join(BASE_DIR, "default.config")

        if os.path.exists(path_camera):
            mvsdk.CameraReadParameterFromFile(
                self._hCamera,
                path_camera
            )       
        elif os.path.exists(path_camera_default):       
            mvsdk.CameraReadParameterFromFile(
                self._hCamera,
                path_camera_default
            )


        # Doc cau hinh camera tu file (neu co) sau khi mo camera thanh cong
        img_size = mvsdk.CameraGetImageResolution(self._hCamera)
        self.width_img.setValue(img_size.iWidth)
        self.height_img.setValue(img_size.iHeight)
        self.offset_x.setValue(img_size.iHOffsetFOV)
        self.offset_y.setValue(img_size.iVOffsetFOV)

        cap = mvsdk.CameraGetCapability(self._hCamera)
        self._mono_camera = (cap.sIspCapacity.bMonoSensor != 0)

        if self._mono_camera:
            mvsdk.CameraSetIspOutFormat(self._hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self._hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        mvsdk.CameraSetTriggerMode(self._hCamera, 1)

        # img_size = mvsdk.CameraGetImageResolution(self._hCamera)
        # img_size.iHOffsetFOV = 0
        # img_size.iVOffsetFOV = 0
        # img_size.iWidthFOV   = cap.sResolutionRange.iWidthMax
        # img_size.iHeightFOV  = cap.sResolutionRange.iHeightMax
        # img_size.iWidth      = cap.sResolutionRange.iWidthMax
        # img_size.iHeight     = cap.sResolutionRange.iHeightMax
        # mvsdk.CameraSetImageResolution(self._hCamera, img_size)

        mvsdk.CameraPlay(self._hCamera)

        frame_buffer_size = (
            cap.sResolutionRange.iWidthMax
            * cap.sResolutionRange.iHeightMax
            * (1 if self._mono_camera else 3)
        )
        self._pFrameBuffer = mvsdk.CameraAlignMalloc(frame_buffer_size, 16)

        print(f"Camera opened: {dev_info.acSn}")
        return True

    def _close_camera(self):
        if self._hCamera is not None:
            try:
                mvsdk.CameraUnInit(self._hCamera)
            except Exception:
                pass
            self._hCamera = None

        if self._pFrameBuffer is not None:
            try:
                mvsdk.CameraAlignFree(self._pFrameBuffer)
            except Exception:
                pass
            self._pFrameBuffer = None

    # ------------------------------------------------------------------
    # Trigger & capture
    # ------------------------------------------------------------------

    @staticmethod
    def _snap_to_step(value: int, step: int, minimum: int, maximum: int) -> int:
        rounded = round(value / step) * step
        return max(minimum, min(maximum, rounded))

    def _read_and_fix_spinbox(self, spinbox, step: int, minimum: int, maximum: int) -> int:
        fixed = self._snap_to_step(spinbox.value(), step, minimum, maximum)
        if fixed != spinbox.value():
            spinbox.setValue(fixed)
        return fixed

    def _get_img_params(self) -> tuple[int, int, int, int]:
        w  = self._read_and_fix_spinbox(self.width_img,  step=16, minimum=16, maximum=2448)
        h  = self._read_and_fix_spinbox(self.height_img, step=4,  minimum=4,  maximum=2048)
        ox = self._read_and_fix_spinbox(self.offset_x,   step=2,  minimum=0,  maximum=2448 - w)
        oy = self._read_and_fix_spinbox(self.offset_y,   step=2,  minimum=0,  maximum=2048 - h)
        return w, h, ox, oy

    def _on_execute_trigger_clicked(self):
        """Chụp ảnh và hiển thị — luôn chạy trên UI thread (dù gọi từ button hay signal)."""
        if self._hCamera is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có camera nào được mở!")
            return

        width, height, offset_x, offset_y = self._get_img_params()

        try:
            img_size = mvsdk.CameraGetImageResolution(self._hCamera)
            img_size.iHOffsetFOV = offset_x
            img_size.iVOffsetFOV = offset_y
            img_size.iWidthFOV   = width
            img_size.iHeightFOV  = height
            img_size.iWidth      = width
            img_size.iHeight     = height
            mvsdk.CameraSetImageResolution(self._hCamera, img_size)

            mvsdk.CameraSoftTrigger(self._hCamera)

            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self._hCamera, 2000)
            mvsdk.CameraImageProcess(self._hCamera, pRawData, self._pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self._hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self._pFrameBuffer, FrameHead, 1)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self._pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                FrameHead.iHeight,
                FrameHead.iWidth,
                1 if self._mono_camera else 3,
            )

            target_size = self.img_cam.size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                return

            qt_image = convert_cv_qt(frame)
            scaled = qt_image.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.img_cam.setPixmap(scaled)

        except mvsdk.CameraException as e:
            QMessageBox.critical(self, "Lỗi chụp ảnh", f"Capture error:\n{e.message}")

    # ------------------------------------------------------------------
    # Save camera config
    # ------------------------------------------------------------------

    def _on_save_camera_config_clicked(self):
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên project!")
            self.text_name_project.setFocus()
            return

        invalid_chars = r'\/:*?"<>|'
        if any(ch in project_name for ch in invalid_chars):
            QMessageBox.warning(self, "Cảnh báo",
                                f"Tên project không được chứa các ký tự: {invalid_chars}")
            return

        if self._hCamera is None or self._current_dev_info is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có camera nào được mở!")
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir   = os.path.join(projects_root, project_name)

        link_name = self._current_dev_info.acSn
        if isinstance(link_name, bytes):
            link_name = link_name.decode()

        config_filename = f"{link_name}.config"
        path = os.path.join(project_dir, "camera_config", config_filename)

        try:
            mvsdk.CameraSaveParameterToFile(self._hCamera, path)
            QMessageBox.information(self, "Thành công",
                                    f"Đã lưu cấu hình camera ra file:\n{config_filename}")
        except mvsdk.CameraException as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu cấu hình:\n{e.message}")


    # ------------------------------------------------------------------
    # Project list
    # ------------------------------------------------------------------

    def _setup_project_list(self):
        self._project_list_model = QStringListModel()
        self.list_project.setModel(self._project_list_model)
        self._refresh_project_list()

    def _refresh_project_list(self):
        projects_root = os.path.join(BASE_DIR, "projects")
        if not os.path.isdir(projects_root):
            self._project_list_model.setStringList([])
            return
        entries = sorted(
            e for e in os.listdir(projects_root)
            if os.path.isdir(os.path.join(projects_root, e))
        )
        self._project_list_model.setStringList(entries)

    def _on_project_selected(self, selected, _):
        indexes = selected.indexes()
        if not indexes:
            return
        name = self._project_list_model.data(indexes[0])
        self.text_name_project.setText(name)

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def _on_add_project_clicked(self):
        self.text_name_project.setEnabled(True)
        self.btn_create_project.setEnabled(True)
        self.text_name_project.setFocus()
        self.text_name_project.clear()

    def _on_create_project_clicked(self):
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên project!")
            self.text_name_project.setFocus()
            return

        invalid_chars = r'\/:*?"<>|'
        if any(ch in project_name for ch in invalid_chars):
            QMessageBox.warning(self, "Cảnh báo",
                                f"Tên project không được chứa các ký tự: {invalid_chars}")
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir   = os.path.join(projects_root, project_name)

        if os.path.exists(project_dir):
            QMessageBox.warning(self, "Cảnh báo", f"Project '{project_name}' đã tồn tại!")
            return

        try:
            self._create_project_structure(project_dir, project_name)
            QMessageBox.information(self, "Thành công",
                                    f"Project '{project_name}' đã được tạo thành công!\n"
                                    f"Đường dẫn: {project_dir}")
            self._setup_initial_state()
            self._refresh_project_list()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo project:\n{str(e)}")

    def _create_project_structure(self, project_dir: str, project_name: str):
        sub_dirs = ["camera_config", "goods", "template_level", "memory_bank"]
        os.makedirs(project_dir, exist_ok=True)
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(project_dir, sub_dir), exist_ok=True)

        project_info = {
            "project_name": project_name,
            "created_at":   self._get_current_timestamp(),
            "description":  "",
            "camera_config": {"cameras": []},
            "settings": {
                "goods_dir":        "goods",
                "memory_bank_dir":  "memory_bank",
                "camera_config_dir":"camera_config",
                "threshold":        None,
                "ratio_liquid":     None,
            },
        }

        json_path = os.path.join(project_dir, "project_info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(project_info, f, ensure_ascii=False, indent=4)

    def _on_delete_project_clicked(self):
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có project nào được chọn!")
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir   = os.path.join(projects_root, project_name)

        if not os.path.exists(project_dir):
            self._refresh_project_list()
            return

        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc muốn xóa project '{project_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            import shutil
            shutil.rmtree(project_dir)
            self.text_name_project.clear()
            self._refresh_project_list()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa project:\n{str(e)}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.stop_all_threads()
        super().closeEvent(event)

    def stop_all_threads(self):
        self.stop_event.set()

        if self._trigger_worker and self._trigger_worker.isRunning():
            self._trigger_worker.wait(2000)

        if self.thread_trigger_camera and self.thread_trigger_camera.is_alive():
            self.thread_trigger_camera.join(timeout=2)

        self._close_camera()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_timestamp() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")