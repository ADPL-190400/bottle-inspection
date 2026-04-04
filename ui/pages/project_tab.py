# from PyQt6 import uic, QtWidgets
# from PyQt6.QtCore import Qt, pyqtSignal, QThread
# from PyQt6.QtWidgets import QMessageBox
# import queue
# import numpy as np
# from core.path_manager import BASE_DIR
# import os
# import threading
# import time
# import json
# import cv2
# from hardware.camera import mvsdk
# from utils.ultis import convert_cv_qt


# BATCH_COLLECTION_TIMEOUT = 1/30


# class ProjectTab(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         ui_path = os.path.join(BASE_DIR, "ui", 'ui', "project_tab.ui")
#         uic.loadUi(ui_path, self)

#         self._setup_initial_state()
#         self._connect_signals()
#         self._list_camera()

#     def _setup_initial_state(self):
#         """Khởi tạo trạng thái ban đầu của các widget."""
#         self.text_name_project.setEnabled(False)
#         self.btn_create_project.setEnabled(False)
#         # self.text_name_project.clear()

#     def _connect_signals(self):
#         """Kết nối các signal với slot."""
#         self.btn_add_project.clicked.connect(self._on_add_project_clicked)
#         self.btn_create_project.clicked.connect(self._on_create_project_clicked)

#     def _on_add_project_clicked(self):
#         """Khi nhấn btn_add_project: enable QLineEdit và btn_create_project."""
#         self.text_name_project.setEnabled(True)
#         self.btn_create_project.setEnabled(True)
#         self.text_name_project.setFocus()
#         self.text_name_project.clear()

#     def _on_create_project_clicked(self):
#         """Khi nhấn btn_create_project: tạo thư mục project và file JSON."""
#         project_name = self.text_name_project.text().strip()

#         if not project_name:
#             QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên project!")
#             self.text_name_project.setFocus()
#             return

#         # Kiểm tra tên hợp lệ (không chứa ký tự đặc biệt)
#         invalid_chars = r'\/:*?"<>|'
#         if any(ch in project_name for ch in invalid_chars):
#             QMessageBox.warning(
#                 self, "Cảnh báo",
#                 f"Tên project không được chứa các ký tự: {invalid_chars}"
#             )
#             return

#         # Đường dẫn gốc chứa các project
#         projects_root = os.path.join(BASE_DIR, "projects")
#         project_dir = os.path.join(projects_root, project_name)

#         if os.path.exists(project_dir):
#             QMessageBox.warning(
#                 self, "Cảnh báo",
#                 f"Project '{project_name}' đã tồn tại!"
#             )
#             return

#         try:
#             self._create_project_structure(project_dir, project_name)
#             QMessageBox.information(
#                 self, "Thành công",
#                 f"Project '{project_name}' đã được tạo thành công!\nĐường dẫn: {project_dir}"
#             )
#             # Reset trạng thái sau khi tạo xong
#             self._setup_initial_state()

#         except Exception as e:
#             QMessageBox.critical(
#                 self, "Lỗi",
#                 f"Không thể tạo project:\n{str(e)}"
#             )

#     def _create_project_structure(self, project_dir: str, project_name: str):
#         """Tạo cấu trúc thư mục và file JSON cho project."""
#         # Các thư mục con cần tạo
#         sub_dirs = [
#             "camera_config",
#             "goods",
#             "template_level",
#             "memory_bank",
#         ]

#         # Tạo thư mục gốc của project
#         os.makedirs(project_dir, exist_ok=True)

#         # Tạo các thư mục con
#         for sub_dir in sub_dirs:
#             os.makedirs(os.path.join(project_dir, sub_dir), exist_ok=True)

#         # Tạo file JSON chứa thông tin project
#         project_info = {
#             "project_name": project_name,
#             "created_at": self._get_current_timestamp(),
#             "description": "",
#             "camera_config": {
#                 "cameras": []
#             },
#             "settings": {
#                 "goods_dir": "goods",
#                 "memory_bank_dir": "memory_bank",
#                 "camera_config_dir": "camera_config",
#                 "threshold": None,
#                 "ratio_liquid": None,
#             }
#         }

#         json_path = os.path.join(project_dir, "project_info.json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(project_info, f, ensure_ascii=False, indent=4)


#     def _list_camera(self):
#         DevList = mvsdk.CameraEnumerateDevice()
#         if len(DevList) == 0:
#             print("No camera found")
#             return
        

#         for DevInfo in DevList:
#             print(f"name: {DevInfo.acSn}")
       

#     @staticmethod
#     def _get_current_timestamp() -> str:
#         """Trả về timestamp hiện tại dạng chuỗi."""
#         from datetime import datetime
#         return datetime.now().strftime("%Y-%m-%d %H:%M:%S")











# coding=utf-8
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import QMessageBox
import queue
import numpy as np
from core.path_manager import BASE_DIR
import os
import threading
import time
import json
import platform
import cv2
from hardware.camera import mvsdk
from utils.ultis import convert_cv_qt


BATCH_COLLECTION_TIMEOUT = 1 / 30


class ProjectTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", "ui", "project_tab.ui")
        uic.loadUi(ui_path, self)

        # Camera state
        self._dev_list = []          # list of DevInfo objects
        self._current_dev_info = None
        self._hCamera = None
        self._pFrameBuffer = None
        self._mono_camera = False

        self._setup_initial_state()
        self._connect_signals()
        self._list_camera()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _setup_initial_state(self):
        """Khởi tạo trạng thái ban đầu của các widget."""
        self.text_name_project.setEnabled(False)
        self.btn_create_project.setEnabled(False)
        self.btn_execute_trigger.setEnabled(False)

    def _connect_signals(self):
        """Kết nối các signal với slot."""
        self.btn_add_project.clicked.connect(self._on_add_project_clicked)
        self.btn_create_project.clicked.connect(self._on_create_project_clicked)
        self.devices_online.currentIndexChanged.connect(self._on_device_selected)
        self.btn_execute_trigger.clicked.connect(self._on_execute_trigger_clicked)
        self.btn_save_camera_config.clicked.connect(self._on_save_camera_config_clicked)

        # Auto-round spinbox values khi người dùng chỉnh
        self.width_img.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.width_img, step=16, minimum=16, maximum=2448))
        self.height_img.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.height_img, step=4, minimum=4, maximum=2048))
        self.offset_x.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.offset_x, step=2, minimum=0, maximum=2448))
        self.offset_y.editingFinished.connect(
            lambda: self._read_and_fix_spinbox(self.offset_y, step=2, minimum=0, maximum=2048))

    # ------------------------------------------------------------------
    # Camera enumeration
    # ------------------------------------------------------------------

    def _list_camera(self):
        """Liệt kê các camera và điền vào ComboBox."""
        self._dev_list = mvsdk.CameraEnumerateDevice()
        self.devices_online.clear()

        if len(self._dev_list) == 0:
            print("No camera found")
            self.devices_online.addItem("-- No camera found --")
            self.btn_execute_trigger.setEnabled(False)
            return

        for dev_info in self._dev_list:
            link_name = dev_info.acSn.decode() if isinstance(dev_info.acSn, bytes) else dev_info.acSn
            self.devices_online.addItem(link_name)

        # Tự động chọn camera đầu tiên
        self.devices_online.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    def _on_device_selected(self, index: int):
        """Khi người dùng chọn camera từ ComboBox."""
        # Đóng camera cũ nếu đang mở
        self._close_camera()

        if index < 0 or index >= len(self._dev_list):
            self._current_dev_info = None
            self.btn_execute_trigger.setEnabled(False)
            return

        self._current_dev_info = self._dev_list[index]
        print(f"Selected camera: {self._current_dev_info.acSn}")

        # Mở camera mới
        success = self._open_camera(self._current_dev_info)
        self.btn_execute_trigger.setEnabled(success)

    # ------------------------------------------------------------------
    # Camera open / close
    # ------------------------------------------------------------------

    def _open_camera(self, dev_info) -> bool:
        """Mở camera với DevInfo cho trước. Trả về True nếu thành công."""
        try:
            self._hCamera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as e:
            QMessageBox.critical(self, "Lỗi camera", f"CameraInit Failed:\n{e.message}")
            self._hCamera = None
            return False
        
        mvsdk.CameraReadParameterFromFile(self._hCamera, "/home/via/Documents/bottle-inspection-main/default.config")
        cap = mvsdk.CameraGetCapability(self._hCamera)
        self._mono_camera = (cap.sIspCapacity.bMonoSensor != 0)

        if self._mono_camera:
            mvsdk.CameraSetIspOutFormat(self._hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self._hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Trigger mode (software trigger)
        mvsdk.CameraSetTriggerMode(self._hCamera, 1)

        # Cấu hình độ phân giải đầy đủ
        img_size = mvsdk.CameraGetImageResolution(self._hCamera)
        img_size.iHOffsetFOV = 0
        img_size.iVOffsetFOV = 0
        img_size.iWidthFOV = cap.sResolutionRange.iWidthMax
        img_size.iHeightFOV = cap.sResolutionRange.iHeightMax
        img_size.iWidth = cap.sResolutionRange.iWidthMax
        img_size.iHeight = cap.sResolutionRange.iHeightMax
        mvsdk.CameraSetImageResolution(self._hCamera, img_size)

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
        """Đóng camera hiện tại nếu đang mở."""
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
        """Làm tròn value về bội số gần nhất của step, clamp vào [minimum, maximum]."""
        rounded = round(value / step) * step
        return max(minimum, min(maximum, rounded))

    def _read_and_fix_spinbox(self, spinbox, step: int, minimum: int, maximum: int) -> int:
        """Đọc spinbox, auto-round nếu cần, cập nhật lại UI, trả về giá trị đã làm tròn."""
        fixed = self._snap_to_step(spinbox.value(), step, minimum, maximum)
        if fixed != spinbox.value():
            spinbox.setValue(fixed)
        return fixed

    def _get_img_params(self) -> tuple[int, int, int, int]:
        """Lấy (width, height, offset_x, offset_y) từ spinbox, tự làm tròn nếu sai."""
        w  = self._read_and_fix_spinbox(self.width_img,  step=16, minimum=16,  maximum=2448)
        h  = self._read_and_fix_spinbox(self.height_img, step=4,  minimum=4,   maximum=2048)
        ox = self._read_and_fix_spinbox(self.offset_x,   step=2,  minimum=0,   maximum=2448-w)
        oy = self._read_and_fix_spinbox(self.offset_y,   step=2,  minimum=0,   maximum=2048-h)
        return w, h, ox, oy

    def _on_execute_trigger_clicked(self):
        """Khi nhấn btn_execute_trigger: software trigger và hiển thị ảnh."""
        if self._hCamera is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có camera nào được mở!")
            return

        width, height, offset_x, offset_y = self._get_img_params()

        try:
            # Áp dụng ROI từ spinbox trước khi trigger
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
            # self.img_cam.setScaledContents(True)

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
            QMessageBox.warning(
                self, "Cảnh báo",
                f"Tên project không được chứa các ký tự: {invalid_chars}"
            )
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir = os.path.join(projects_root, project_name)

        """Lưu cấu hình camera ra file <acSn>.config."""
        if self._hCamera is None or self._current_dev_info is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có camera nào được mở!")
            return

        link_name = self._current_dev_info.acSn
        if isinstance(link_name, bytes):
            link_name = link_name.decode()

        config_filename = f"{link_name}.config"
        path = os.path.join(project_dir,"camera_config",config_filename)

        try:
            mvsdk.CameraSaveParameterToFile(self._hCamera, path)
            QMessageBox.information(
                self, "Thành công",
                f"Đã lưu cấu hình camera ra file:\n{config_filename}"
            )
        except mvsdk.CameraException as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu cấu hình:\n{e.message}")

    # ------------------------------------------------------------------
    # Project management (giữ nguyên)
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
            QMessageBox.warning(
                self, "Cảnh báo",
                f"Tên project không được chứa các ký tự: {invalid_chars}"
            )
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir = os.path.join(projects_root, project_name)

        if os.path.exists(project_dir):
            QMessageBox.warning(self, "Cảnh báo", f"Project '{project_name}' đã tồn tại!")
            return

        try:
            self._create_project_structure(project_dir, project_name)
            QMessageBox.information(
                self, "Thành công",
                f"Project '{project_name}' đã được tạo thành công!\nĐường dẫn: {project_dir}"
            )
            self._setup_initial_state()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo project:\n{str(e)}")

    def _create_project_structure(self, project_dir: str, project_name: str):
        sub_dirs = ["camera_config", "goods", "template_level", "memory_bank"]
        os.makedirs(project_dir, exist_ok=True)
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(project_dir, sub_dir), exist_ok=True)

        project_info = {
            "project_name": project_name,
            "created_at": self._get_current_timestamp(),
            "description": "",
            "camera_config": {"cameras": []},
            "settings": {
                "goods_dir": "goods",
                "memory_bank_dir": "memory_bank",
                "camera_config_dir": "camera_config",
                "threshold": None,
                "ratio_liquid": None,
            },
        }

        json_path = os.path.join(project_dir, "project_info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(project_info, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Đảm bảo camera được đóng khi widget bị hủy."""
        self._close_camera()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_timestamp() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")