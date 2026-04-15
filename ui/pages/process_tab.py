from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMetaObject, Q_ARG
import queue
import numpy as np
from pathlib import Path
from core.path_manager import BASE_DIR
import os
import time
import json
from utils.ultis import convert_cv_qt
from core.pipeline_maneger import PipelineManager


BATCH_COLLECTION_TIMEOUT = 1 / 30
PROJECTS_ROOT = os.path.join(BASE_DIR, "projects")
PROJECT_INFO_FILENAME = "project_info.json"


class ProcessTab(QtWidgets.QWidget):
    # Signal thread-safe để nhận callback từ pipeline (thread khác)
    _batch_result_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        ui_path = os.path.join(BASE_DIR, "ui", "ui", "process_tab.ui")
        uic.loadUi(ui_path, self)

        self.btn_play.clicked.connect(self.load_project_json)
        self.btn_stop.clicked.connect(self.stop_all_threads)

        self.is_playing       = False
        self.pipeline_manager = None
        self.frame_updater    = None

        # ── Inspection counters ───────────────────────────────────────────── #
        self._inspected_count = 0
        self._rejected_count  = 0
        self._reset_counters()

        # Kết nối signal (từ pipeline thread) → slot (GUI thread)
        self._batch_result_signal.connect(self.update_classification)

        # ── name_project ComboBox ─────────────────────────────────────────── #
        self._populate_project_combo()
        self.name_project.currentIndexChanged.connect(self._reset_counters)

        self.show_queue = {
            str(cam_id): queue.Queue(maxsize=1) for cam_id in range(1, 6)
        }

    # ----------------------------------------------------------------------- #
    #  COUNTER HELPERS                                                         #
    # ----------------------------------------------------------------------- #
    def _reset_counters(self):
        self._inspected_count = 0
        self._rejected_count  = 0
        self._update_counter_labels()

    def _update_counter_labels(self):
        if hasattr(self, "inspectedCountLabel"):
            self.inspectedCountLabel.setText(str(self._inspected_count))
        if hasattr(self, "rejectedCountLabel"):
            self.rejectedCountLabel.setText(str(self._rejected_count))
        if hasattr(self, "defectRateLabel"):
            if self._inspected_count > 0:
                rate = self._rejected_count / self._inspected_count * 100
                self.defectRateLabel.setText(f"{rate:.1f}%")
            else:
                self.defectRateLabel.setText("0.0%")

    # ----------------------------------------------------------------------- #
    #  PROJECT HELPERS                                                         #
    # ----------------------------------------------------------------------- #
    def _get_projects_root(self) -> str:
        return PROJECTS_ROOT

    def _populate_project_combo(self):
        self.name_project.blockSignals(True)
        self.name_project.clear()
        root = self._get_projects_root()
        if os.path.isdir(root):
            entries = sorted(
                e for e in os.listdir(root)
                if os.path.isdir(os.path.join(root, e))
            )
            self.name_project.addItems(entries)
        else:
            print(f"[GetData] ⚠ Thư mục project không tồn tại: {root}")
        self.name_project.blockSignals(False)

    def _get_project_root(self) -> Path | None:
        name = self.name_project.currentText().strip()
        if not name:
            return None
        return Path(self._get_projects_root()) / name

    def load_project_json(self):
        if self.is_playing:
            print("Pipeline already running")
            return
        print("Load project yaml")

        project_root = Path(self._get_project_root())
        json_path = project_root / PROJECT_INFO_FILENAME

        if not json_path.exists():
            print(f"⚠️  Không tìm thấy {json_path}")
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            infor_project = json.load(f)

        self._reset_counters()
        self.is_playing = True

        self.frame_updater = UpdateFrameThread(self.show_queue)
        self.frame_updater.frame_ready.connect(self.update_frame)
        self.frame_updater.start()

        # Truyền callback vào pipeline — gọi đúng 1 lần / trigger
        self.pipeline_manager = PipelineManager(
            show_queue            = self.show_queue,
            infor_project         = infor_project,
            batch_result_callback = self._on_batch_result,
        )
        self.pipeline_manager.start()

    # ----------------------------------------------------------------------- #
    def _on_batch_result(self, batch_ok: bool):
        """Được gọi từ pipeline thread → emit signal để chuyển sang GUI thread."""
        self._batch_result_signal.emit(batch_ok)

    # ----------------------------------------------------------------------- #
    def stop_all_threads(self):
        if not self.is_playing:
            return
        try:
            print("Stopping all threads...")

            if self.pipeline_manager:
                self.pipeline_manager.stop()
                self.pipeline_manager.join()
                self.pipeline_manager = None

            if self.frame_updater:
                self.frame_updater.stop()
                self.frame_updater.quit()
                self.frame_updater.wait()
                self.frame_updater = None

            for q in self.show_queue.values():
                with q.mutex:
                    q.queue.clear()

            self.is_playing = False
            print("All threads stopped")

        except Exception as e:
            print("Error stopping threads:", e)

    # ----------------------------------------------------------------------- #
    def update_frame(self, cam_id: str, frame: np.ndarray):
        cam_widget = getattr(self, f"cam_{cam_id}", None)
        if cam_widget is None:
            return
        target_size = cam_widget.size()
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
        """
        Được gọi đúng 1 lần mỗi trigger_id từ pipeline
        = 1 lần chụp = 1 sản phẩm.
        """
        self._inspected_count += 1
        if not is_ok:
            self._rejected_count += 1
        self._update_counter_labels()

        label = "OK" if is_ok else "NG"
        color = "green" if is_ok else "red"
        self.text_result.setText(label)
        self.text_result.setStyleSheet(f"color: {color}; font-weight: bold;")


# =========================================================================== #
#  UPDATE FRAME THREAD  (chỉ còn nhiệm vụ hiển thị frame, không đếm nữa)     #
# =========================================================================== #
class UpdateFrameThread(QThread):
    frame_ready = pyqtSignal(str, np.ndarray)   # (cam_id, frame_bgr)

    def __init__(self, show_queue: dict):
        super().__init__()
        self.show_queue = show_queue
        self._running   = True

    def run(self):
        while self._running:
            for cam_id, q in self.show_queue.items():
                try:
                    frame, _ = q.get_nowait()   # bỏ qua is_ok, pipeline lo rồi
                except queue.Empty:
                    continue
                except (TypeError, ValueError):
                    continue
                self.frame_ready.emit(cam_id, frame)

            time.sleep(BATCH_COLLECTION_TIMEOUT / 2)

    def stop(self):
        self._running = False