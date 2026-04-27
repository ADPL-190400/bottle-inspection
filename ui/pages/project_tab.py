# coding=utf-8
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QThread, QStringListModel, Qt, pyqtSignal
from PyQt6.QtWidgets import QMessageBox
import json
import os
import queue
import threading
import time

from core.path_manager import BASE_DIR
from hardware.camera.generic_camera import (
    GenericCamera,
    load_camera_config,
    save_camera_config,
)
from hardware.gpio.trigger_input_camera import TriggerCamera
from utils.ultis import convert_cv_qt


BATCH_COLLECTION_TIMEOUT = 1 / 30


class TriggerWorker(QThread):
    triggered = pyqtSignal()

    def __init__(self, trigger_queue: queue.Queue, stop_event: threading.Event, parent=None):
        super().__init__(parent)
        self.trigger_queue = trigger_queue
        self.stop_event = stop_event

    def run(self):
        print("[TriggerWorker] Running... waiting for trigger")
        while not self.stop_event.is_set():
            try:
                if self.trigger_queue.empty():
                    time.sleep(0.01)
                    continue
                _ = self.trigger_queue.get_nowait()
                self.triggered.emit()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[TriggerWorker] {e}")


class ProjectTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", "ui", "project_tab.ui")
        uic.loadUi(ui_path, self)

        self._dev_list = []
        self._current_dev_info = None
        self._hCamera = None
        self._camera_info = None

        self.stop_event = threading.Event()
        self.trigger_queue = queue.Queue(maxsize=1)
        self.thread_trigger_camera = None
        self._trigger_worker = None

        self._setup_initial_state()
        self._setup_project_list()
        self._connect_signals()
        self._list_camera()
        self._init_trigger_camera()

    def _setup_initial_state(self):
        self.text_name_project.setEnabled(False)
        self.btn_create_project.setEnabled(False)
        self.btn_execute_trigger.setEnabled(False)

    def _init_trigger_camera(self):
        self.thread_trigger_camera = TriggerCamera(
            self.trigger_queue, stop_event=self.stop_event, offset=3
        )
        self.thread_trigger_camera.start()

        self._trigger_worker = TriggerWorker(self.trigger_queue, self.stop_event, parent=self)
        self._trigger_worker.triggered.connect(self._on_execute_trigger_clicked)
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
            lambda: self._fix_spinbox(self.width_img, "Width", fallback_min=16)
        )
        self.height_img.editingFinished.connect(
            lambda: self._fix_spinbox(self.height_img, "Height", fallback_min=4)
        )
        self.offset_x.editingFinished.connect(
            lambda: self._fix_spinbox(self.offset_x, "OffsetX", fallback_min=0)
        )
        self.offset_y.editingFinished.connect(
            lambda: self._fix_spinbox(self.offset_y, "OffsetY", fallback_min=0)
        )

    def _list_camera(self):
        try:
            self._dev_list = GenericCamera.enumerate_devices()
        except Exception as e:
            self._dev_list = []
            print(f"List camera failed: {e}")

        self.devices_online.clear()

        if not self._dev_list:
            print("No camera found")
            self.devices_online.addItem("-- No camera found --")
            self.btn_execute_trigger.setEnabled(False)
            return

        for dev_info in self._dev_list:
            self.devices_online.addItem(dev_info.device_id)

        self.devices_online.setCurrentIndex(0)

    def _on_device_selected(self, index: int):
        self._close_camera()

        if index < 0 or index >= len(self._dev_list):
            self._current_dev_info = None
            self.btn_execute_trigger.setEnabled(False)
            return

        self._current_dev_info = self._dev_list[index]
        print(f"Selected camera: {self._current_dev_info.device_id}")

        success = self._open_camera(self._current_dev_info)
        self.btn_execute_trigger.setEnabled(success)

    def _camera_config_path(self, project_name: str) -> str:
        return os.path.join(
            BASE_DIR,
            "projects",
            project_name,
            "camera_config",
            f"{GenericCamera.config_key(self._current_dev_info.device_id)}.json",
        )

    def _open_camera(self, dev_info) -> bool:
        try:
            self._hCamera = GenericCamera(dev_info)
        except Exception as e:
            QMessageBox.critical(self, "Loi camera", f"CameraInit Failed:\n{e}")
            self._hCamera = None
            return False

        project_name = self.text_name_project.text().strip()
        if self._hCamera is None or self._current_dev_info is None:
            QMessageBox.warning(self, "Canh bao", "Chua co camera nao duoc mo!")
            return False

        if project_name:
            config = load_camera_config(self._camera_config_path(project_name))
            if config is not None:
                self._hCamera.apply_config(config)

        self._camera_info = self._hCamera.get_camera_info()
        offset_x, offset_y, width, height = self._hCamera.get_region()
        self.width_img.setValue(width)
        self.height_img.setValue(height)
        self.offset_x.setValue(offset_x)
        self.offset_y.setValue(offset_y)
        self.exposure_time.setValue(int(self._hCamera.get_exposure_time()))
        self._apply_spinbox_limits()

        self._hCamera.enable_software_trigger(self._camera_info)
        self._hCamera.start_acquisition()
        print(f"Camera opened: {dev_info.device_id}")
        return True

    def _close_camera(self):
        if self._hCamera is not None:
            try:
                self._hCamera.close()
            except Exception:
                pass
            self._hCamera = None
        self._camera_info = None

    @staticmethod
    def _snap_to_step(value: int, step: int, minimum: int, maximum: int) -> int:
        rounded = round(value / step) * step
        return max(minimum, min(maximum, rounded))

    def _get_feature_meta(self, name: str, fallback_min: int = 0) -> dict:
        if self._camera_info and name in self._camera_info:
            meta = self._camera_info[name]
            return {
                "min": int(meta.get("min", fallback_min)),
                "max": int(meta.get("max", fallback_min)),
                "inc": max(1, int(meta.get("inc", 1))),
            }
        return {"min": fallback_min, "max": max(fallback_min, 1), "inc": 1}

    def _apply_spinbox_limits(self):
        width_meta = self._get_feature_meta("Width", fallback_min=16)
        height_meta = self._get_feature_meta("Height", fallback_min=4)
        offset_x_meta = self._get_feature_meta("OffsetX", fallback_min=0)
        offset_y_meta = self._get_feature_meta("OffsetY", fallback_min=0)

        self.width_img.setRange(width_meta["min"], width_meta["max"])
        self.height_img.setRange(height_meta["min"], height_meta["max"])
        self.offset_x.setRange(offset_x_meta["min"], offset_x_meta["max"])
        self.offset_y.setRange(offset_y_meta["min"], offset_y_meta["max"])

        self.width_img.setSingleStep(width_meta["inc"])
        self.height_img.setSingleStep(height_meta["inc"])
        self.offset_x.setSingleStep(offset_x_meta["inc"])
        self.offset_y.setSingleStep(offset_y_meta["inc"])

    def _fix_spinbox(self, spinbox, feature_name: str, fallback_min: int = 0) -> int:
        meta = self._get_feature_meta(feature_name, fallback_min=fallback_min)
        fixed = self._snap_to_step(spinbox.value(), meta["inc"], meta["min"], meta["max"])
        if fixed != spinbox.value():
            spinbox.setValue(fixed)
        return fixed

    def _get_img_params(self) -> tuple[int, int, int, int]:
        width = self._fix_spinbox(self.width_img, "Width", fallback_min=16)
        height = self._fix_spinbox(self.height_img, "Height", fallback_min=4)

        ox_meta = self._get_feature_meta("OffsetX", fallback_min=0)
        oy_meta = self._get_feature_meta("OffsetY", fallback_min=0)
        width_meta = self._get_feature_meta("Width", fallback_min=16)
        height_meta = self._get_feature_meta("Height", fallback_min=4)

        max_offset_x = max(ox_meta["min"], width_meta["max"] - width)
        max_offset_y = max(oy_meta["min"], height_meta["max"] - height)

        offset_x = self._snap_to_step(self.offset_x.value(), ox_meta["inc"], ox_meta["min"], max_offset_x)
        offset_y = self._snap_to_step(self.offset_y.value(), oy_meta["inc"], oy_meta["min"], max_offset_y)

        if offset_x != self.offset_x.value():
            self.offset_x.setValue(offset_x)
        if offset_y != self.offset_y.value():
            self.offset_y.setValue(offset_y)

        return width, height, offset_x, offset_y

    def _on_execute_trigger_clicked(self):
        if self._hCamera is None:
            QMessageBox.warning(self, "Canh bao", "Chua co camera nao duoc mo!")
            return

        width, height, offset_x, offset_y = self._get_img_params()
        exposure_time = self.exposure_time.value()

        try:
            self._hCamera.set_exposure_time(exposure_time)
            self._hCamera.set_roi(offset_x, offset_y, width, height, self._camera_info)
            self._hCamera.execute_software_trigger()
            frame = self._hCamera.grab_frame(timeout_us=2_000_000, color_order="bgr")
            if frame is None:
                raise RuntimeError("Capture timeout")

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
        except Exception as e:
            QMessageBox.critical(self, "Loi chup anh", f"Capture error:\n{e}")

    def _on_save_camera_config_clicked(self):
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Canh bao", "Vui long nhap ten project!")
            self.text_name_project.setFocus()
            return

        invalid_chars = r'\/:*?"<>|'
        if any(ch in project_name for ch in invalid_chars):
            QMessageBox.warning(
                self,
                "Canh bao",
                f"Ten project khong duoc chua cac ky tu: {invalid_chars}",
            )
            return

        if self._hCamera is None or self._current_dev_info is None:
            QMessageBox.warning(self, "Canh bao", "Chua co camera nao duoc mo!")
            return

        width, height, offset_x, offset_y = self._get_img_params()
        self._hCamera.set_exposure_time(self.exposure_time.value())
        self._hCamera.set_roi(offset_x, offset_y, width, height, self._camera_info)

        path = self._camera_config_path(project_name)
        config = self._hCamera.export_config()
        try:
            save_camera_config(path, config)
            QMessageBox.information(
                self,
                "Thanh cong",
                f"Da luu cau hinh camera ra file:\n{os.path.basename(path)}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Loi", f"Khong the luu cau hinh:\n{e}")

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

    def _on_add_project_clicked(self):
        self.text_name_project.setEnabled(True)
        self.btn_create_project.setEnabled(True)
        self.text_name_project.setFocus()
        self.text_name_project.clear()

    def _on_create_project_clicked(self):
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Canh bao", "Vui long nhap ten project!")
            self.text_name_project.setFocus()
            return

        invalid_chars = r'\/:*?"<>|'
        if any(ch in project_name for ch in invalid_chars):
            QMessageBox.warning(
                self,
                "Canh bao",
                f"Ten project khong duoc chua cac ky tu: {invalid_chars}",
            )
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir = os.path.join(projects_root, project_name)

        if os.path.exists(project_dir):
            QMessageBox.warning(self, "Canh bao", f"Project '{project_name}' da ton tai!")
            return

        try:
            self._create_project_structure(project_dir, project_name)
            QMessageBox.information(
                self,
                "Thanh cong",
                f"Project '{project_name}' da duoc tao thanh cong!\nDuong dan: {project_dir}",
            )
            self._setup_initial_state()
            self._refresh_project_list()
        except Exception as e:
            QMessageBox.critical(self, "Loi", f"Khong the tao project:\n{str(e)}")

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

    def _on_delete_project_clicked(self):
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Canh bao", "Chua co project nao duoc chon!")
            return

        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir = os.path.join(projects_root, project_name)

        if not os.path.exists(project_dir):
            self._refresh_project_list()
            return

        reply = QMessageBox.question(
            self,
            "Xac nhan xoa",
            f"Ban co chac muon xoa project '{project_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            import shutil

            shutil.rmtree(project_dir)
            self.text_name_project.clear()
            self._refresh_project_list()
        except Exception as e:
            QMessageBox.critical(self, "Loi", f"Khong the xoa project:\n{str(e)}")

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

    @staticmethod
    def _get_current_timestamp() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
