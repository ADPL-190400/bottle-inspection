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
import cv2
from utils.ultis import convert_cv_qt


BATCH_COLLECTION_TIMEOUT = 1/30


class ProjectTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", 'ui', "project_tab.ui")
        uic.loadUi(ui_path, self)

        self._setup_initial_state()
        self._connect_signals()

    def _setup_initial_state(self):
        """Khởi tạo trạng thái ban đầu của các widget."""
        self.text_name_project.setEnabled(False)
        self.btn_create_project.setEnabled(False)
        # self.text_name_project.clear()

    def _connect_signals(self):
        """Kết nối các signal với slot."""
        self.btn_add_project.clicked.connect(self._on_add_project_clicked)
        self.btn_create_project.clicked.connect(self._on_create_project_clicked)

    def _on_add_project_clicked(self):
        """Khi nhấn btn_add_project: enable QLineEdit và btn_create_project."""
        self.text_name_project.setEnabled(True)
        self.btn_create_project.setEnabled(True)
        self.text_name_project.setFocus()
        self.text_name_project.clear()

    def _on_create_project_clicked(self):
        """Khi nhấn btn_create_project: tạo thư mục project và file JSON."""
        project_name = self.text_name_project.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên project!")
            self.text_name_project.setFocus()
            return

        # Kiểm tra tên hợp lệ (không chứa ký tự đặc biệt)
        invalid_chars = r'\/:*?"<>|'
        if any(ch in project_name for ch in invalid_chars):
            QMessageBox.warning(
                self, "Cảnh báo",
                f"Tên project không được chứa các ký tự: {invalid_chars}"
            )
            return

        # Đường dẫn gốc chứa các project
        projects_root = os.path.join(BASE_DIR, "projects")
        project_dir = os.path.join(projects_root, project_name)

        if os.path.exists(project_dir):
            QMessageBox.warning(
                self, "Cảnh báo",
                f"Project '{project_name}' đã tồn tại!"
            )
            return

        try:
            self._create_project_structure(project_dir, project_name)
            QMessageBox.information(
                self, "Thành công",
                f"Project '{project_name}' đã được tạo thành công!\nĐường dẫn: {project_dir}"
            )
            # Reset trạng thái sau khi tạo xong
            self._setup_initial_state()

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi",
                f"Không thể tạo project:\n{str(e)}"
            )

    def _create_project_structure(self, project_dir: str, project_name: str):
        """Tạo cấu trúc thư mục và file JSON cho project."""
        # Các thư mục con cần tạo
        sub_dirs = [
            "camera_config",
            "goods",
            "template_level",
            "memory_bank",
        ]

        # Tạo thư mục gốc của project
        os.makedirs(project_dir, exist_ok=True)

        # Tạo các thư mục con
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(project_dir, sub_dir), exist_ok=True)

        # Tạo file JSON chứa thông tin project
        project_info = {
            "project_name": project_name,
            "created_at": self._get_current_timestamp(),
            "description": "",
            "camera_config": {
                "cameras": []
            },
            "settings": {
                "goods_dir": "goods",
                "memory_bank_dir": "memory_bank",
                "camera_config_dir": "camera_config",
                "threshold": None,
                "ratio_liquid": None,
            }
        }

        json_path = os.path.join(project_dir, "project_info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(project_info, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _get_current_timestamp() -> str:
        """Trả về timestamp hiện tại dạng chuỗi."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
