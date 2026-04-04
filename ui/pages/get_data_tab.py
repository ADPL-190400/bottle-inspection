# from PyQt6 import uic, QtWidgets
# from PyQt6.QtCore import QThread, pyqtSignal, QDir, QPoint, QRect, Qt
# from PyQt6.QtGui import QFileSystemModel, QPixmap, QPainter, QPen, QColor, QBrush, QImage
# import queue
# from core.path_manager import BASE_DIR
# import os
# import re
# import threading
# import cv2
# import numpy as np
# from pathlib import Path
# from hardware.camera.batch_camera import BatchCamera
# from training.patchcore_memory_bank import build_bank
# import json


# PROJECTS_ROOT           = os.path.join(BASE_DIR, "projects")
# PROJECT_INFO_FILENAME   = "project_info.json"
# TEMPLATE_POINT_FILENAME = "template_point.json"
# U2NET_ENGINE_PATH       = os.path.join(BASE_DIR, "models/remove_bg/u2netp.trt")


# # =========================================================================== #
# #  BUILD WORKER                                                                #
# # =========================================================================== #
# class BuildBankWorker(QThread):
#     finished = pyqtSignal(bool, str)

#     def __init__(self, goods_dir: Path, output_dir: Path):
#         super().__init__()
#         self.goods_dir  = goods_dir
#         self.output_dir = output_dir

#     def run(self):
#         try:
#             build_bank(self.goods_dir, self.output_dir)
#             self.finished.emit(True, f"✅ Build hoàn tất.\n{self.output_dir / 'memory_bank.pt'}")
#         except Exception as e:
#             self.finished.emit(False, f"❌ Build thất bại: {e}")


# # =========================================================================== #
# #  LIQUID SETUP WORKER                                                         #
# #  Tạo U2NetSegmentor riêng (cùng code như BodyWorker, dùng chung engine file) #
# # =========================================================================== #
# class LiquidSetupWorker(QThread):
#     """
#     Chạy trong QThread riêng để không block UI:
#       1. Tạo U2NetSegmentor (cùng engine file với BodyWorker)
#       2. attach() → get_mask() → detach()
#       3. LiquidLevelDetector.detect(img, mask) → baseline fill_ratio
#       4. emit detect_done

#     Sau khi user nhập tolerance → gọi save_final() (từ UI thread, an toàn).
#     """
#     # (success, fill_ratio, display_img_bgr, error_msg)
#     detect_done = pyqtSignal(bool, float, object, str)

#     def __init__(self, img_bgr: np.ndarray,
#                  template_gray: np.ndarray,
#                  x: int, w: int, h: int,
#                  project_root: Path):
#         super().__init__()
#         self.img_bgr       = img_bgr
#         self.template_gray = template_gray
#         self.x             = x
#         self.w             = w
#         self.h             = h
#         self.project_root  = project_root

#         # Được set sau khi detect xong, dùng trong save_final()
#         self._baseline    = None
#         self._detector    = None
#         self._object_mask = None

#     # ------------------------------------------------------------------ #
#     def run(self):
#         from core.liquid_level import LiquidLevelDetector

#         detector = LiquidLevelDetector(self.project_root)

#         # Lưu config tạm để detect() có thể chạy
#         detector.save_setup(
#             self.template_gray,
#             x=self.x, w=self.w, h=self.h,
#             baseline=50.0, tolerance=10.0,
#         )

#         # ── Lấy mask từ U2Net (tương tự cách BodyWorker làm) ─────────────
#         object_mask = None
#         if os.path.exists(U2NET_ENGINE_PATH):
#             try:
#                 from core.u2net_segmentor import U2NetSegmentor
#                 seg = U2NetSegmentor(U2NET_ENGINE_PATH)   # instance riêng, cùng engine
#                 seg.attach()
#                 object_mask = seg.get_mask(self.img_bgr)
#                 seg.detach()
#                 print("[LiquidSetup] ✅ U2Net mask OK.")
#             except Exception as e:
#                 print(f"[LiquidSetup] ⚠️  U2Net thất bại: {e} — detect không có mask")
#         else:
#             print(f"[LiquidSetup] ⚠️  Không tìm thấy engine — detect không có mask")

#         # ── Detect → lấy baseline ─────────────────────────────────────────
#         try:
#             result = detector.detect(self.img_bgr, object_mask=object_mask)
#         except Exception as e:
#             self.detect_done.emit(False, 0.0, None, f"detect() lỗi: {e}")
#             return

#         if result is None:
#             self.detect_done.emit(
#                 False, 0.0, None,
#                 "Không tìm được mực nước.\nHãy chọn lại vùng ROI chính xác hơn."
#             )
#             return

#         # Lưu lại để save_final() dùng
#         self._baseline    = result["fill_ratio"]
#         self._detector    = detector
#         self._object_mask = object_mask

#         self.detect_done.emit(True, self._baseline, result["display_img"], "")

#     # ------------------------------------------------------------------ #
#     def save_final(self, tolerance: float) -> np.ndarray | None:
#         """
#         Gọi từ UI thread sau khi user nhập tolerance.
#         Lưu config chính thức + chạy detect lần cuối để có display_img đúng.
#         Trả về display_img (BGR numpy) hoặc None.
#         """
#         if self._detector is None or self._baseline is None:
#             return None

#         self._detector.save_setup(
#             self.template_gray,
#             x=self.x, w=self.w, h=self.h,
#             baseline=self._baseline,
#             tolerance=tolerance,
#         )
#         result = self._detector.detect(self.img_bgr, object_mask=self._object_mask)
#         return result["display_img"] if result else None


# # =========================================================================== #
# #  GET DATA TAB                                                                #
# # =========================================================================== #
# class GetDataTab(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         ui_path = os.path.join(BASE_DIR, "ui", "ui", "get_data_tab.ui")
#         uic.loadUi(ui_path, self)

#         self.btn_start_get_data.clicked.connect(self.start_get_data)
#         self.btn_stop_get_data.clicked.connect(self.stop_get_data)
#         self.build_memory_bank.clicked.connect(self.build_memory)
#         self.btn_select_template.clicked.connect(self._on_select_template)
#         self.btn_cancle_select.clicked.connect(self._on_cancel_select)
#         self.btn_setup_liquid.clicked.connect(self._on_setup_liquid)

#         self.data_manager        = None
#         self.is_initialized      = False
#         self.is_saving           = False
#         self._build_worker       = None
#         self._liquid_worker: LiquidSetupWorker | None = None

#         self._select_mode        = None
#         self._template_point_1   = None
#         self._template_point_2   = None
#         self._current_image_path = None

#         self.img_show.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.img_show.clicked.connect(self._on_image_clicked)

#         self._fs_model = QFileSystemModel()
#         self._fs_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
#         self.treeView_img.setModel(self._fs_model)
#         self.treeView_img.setColumnHidden(1, True)
#         self.treeView_img.setColumnHidden(2, True)
#         self.treeView_img.setColumnHidden(3, True)
#         self.treeView_img.setHeaderHidden(False)
#         self.treeView_img.selectionModel().selectionChanged.connect(
#             self._on_tree_selection_changed
#         )

#         self._populate_project_combo()
#         self.name_project.currentIndexChanged.connect(self._on_project_changed)
#         self._update_tree_from_combo()
#         self._update_select_buttons_state()

#     # ======================================================================= #
#     #  PROJECT HELPERS                                                         #
#     # ======================================================================= #
#     def _get_projects_root(self) -> str:
#         return PROJECTS_ROOT

#     def _populate_project_combo(self):
#         self.name_project.blockSignals(True)
#         self.name_project.clear()
#         root = self._get_projects_root()
#         if os.path.isdir(root):
#             entries = sorted(e for e in os.listdir(root)
#                              if os.path.isdir(os.path.join(root, e)))
#             self.name_project.addItems(entries)
#         self.name_project.blockSignals(False)

#     def _on_project_changed(self, _):
#         self._update_tree_from_combo()

#     def _update_tree_from_combo(self):
#         goods = self._get_goods_path()
#         if goods:
#             self._set_tree_root(str(goods))

#     def _get_project_root(self) -> Path | None:
#         name = self.name_project.currentText().strip()
#         return Path(self._get_projects_root()) / name if name else None

#     def _get_goods_path(self) -> Path | None:
#         pr = self._get_project_root()
#         if pr is None:
#             return None
#         g = pr / "goods"
#         g.mkdir(parents=True, exist_ok=True)
#         return g

#     def _get_memory_bank_dir(self) -> Path | None:
#         pr = self._get_project_root()
#         if pr is None:
#             return None
#         d = pr / "memory_bank"
#         d.mkdir(parents=True, exist_ok=True)
#         return d

#     def _get_template_point_path(self) -> Path | None:
#         pr = self._get_project_root()
#         return pr / TEMPLATE_POINT_FILENAME if pr else None

#     # ======================================================================= #
#     #  TREE VIEW                                                               #
#     # ======================================================================= #
#     def _set_tree_root(self, path: str):
#         idx = self._fs_model.setRootPath(path)
#         self.treeView_img.setRootIndex(idx)
#         self.treeView_img.expandAll()

#     def _set_data_buttons_enabled(self, enabled: bool):
#         self.btn_start_get_data.setEnabled(enabled)
#         self.btn_stop_get_data.setEnabled(enabled)
#         self.build_memory_bank.setEnabled(enabled)

#     def _on_tree_selection_changed(self, selected, _):
#         indexes = selected.indexes()
#         if not indexes:
#             return
#         file_path = self._fs_model.filePath(indexes[0])
#         if not os.path.isfile(file_path):
#             return
#         if os.path.splitext(file_path)[1].lower() not in \
#                 (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
#             return
#         self._current_image_path = file_path
#         self._select_mode        = None
#         self._template_point_1   = None
#         self._template_point_2   = None
#         self._update_select_buttons_state()
#         self._display_image(file_path)

#     # ======================================================================= #
#     #  DISPLAY HELPERS                                                         #
#     # ======================================================================= #
#     def _get_scale_offset(self, orig_w, orig_h):
#         lw, lh   = self.img_show.width(), self.img_show.height()
#         scale    = min(lw / orig_w, lh / orig_h)
#         sw, sh   = int(orig_w * scale), int(orig_h * scale)
#         ox, oy   = (lw - sw) // 2, (lh - sh) // 2
#         return scale, sw, sh, ox, oy

#     def _display_image(self, file_path: str,
#                        point1: QPoint | None = None,
#                        point2: QPoint | None = None):
#         pm = QPixmap(file_path)
#         if pm.isNull():
#             return
#         ow, oh = pm.width(), pm.height()
#         scale, sw, sh, ox, oy = self._get_scale_offset(ow, oh)
#         spm = pm.scaled(sw, sh,
#                         Qt.AspectRatioMode.KeepAspectRatio,
#                         Qt.TransformationMode.SmoothTransformation)
#         if point1 is not None:
#             px1, py1 = int(point1.x() - ox), int(point1.y() - oy)
#             p = QPainter(spm)
#             p.setRenderHint(QPainter.RenderHint.Antialiasing)
#             if point2 is None:
#                 p.setPen(QPen(QColor(0, 255, 0), 2))
#                 p.setBrush(Qt.BrushStyle.NoBrush)
#                 p.drawEllipse(QPoint(px1, py1), 6, 6)
#                 p.drawLine(px1 - 10, py1, px1 + 10, py1)
#                 p.drawLine(px1, py1 - 10, px1, py1 + 10)
#             else:
#                 px2, py2 = int(point2.x() - ox), int(point2.y() - oy)
#                 rect = QRect(QPoint(px1, py1), QPoint(px2, py2)).normalized()
#                 p.setBrush(QBrush(QColor(0, 255, 0, 40)))
#                 p.setPen(QPen(QColor(0, 255, 0), 2))
#                 p.drawRect(rect)
#                 p.setBrush(QBrush(QColor(0, 255, 0)))
#                 p.setPen(Qt.PenStyle.NoPen)
#                 p.drawEllipse(QPoint(px1, py1), 4, 4)
#                 p.drawEllipse(QPoint(px2, py2), 4, 4)
#             p.end()
#         self.img_show.setPixmap(spm)

#     def _display_cv2_image(self, img_bgr: np.ndarray):
#         rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
#         pm    = QPixmap.fromImage(qimg)
#         scaled = pm.scaled(self.img_show.width(), self.img_show.height(),
#                            Qt.AspectRatioMode.KeepAspectRatio,
#                            Qt.TransformationMode.SmoothTransformation)
#         self.img_show.setPixmap(scaled)

#     # ======================================================================= #
#     #  COORDINATE CONVERSION                                                   #
#     # ======================================================================= #
#     def _label_to_img(self, lp: QPoint) -> tuple[int, int]:
#         if not self._current_image_path:
#             return lp.x(), lp.y()
#         pm = QPixmap(self._current_image_path)
#         if pm.isNull():
#             return lp.x(), lp.y()
#         scale, _, _, ox, oy = self._get_scale_offset(pm.width(), pm.height())
#         rx = max(0.0, min((lp.x() - ox) / scale, float(pm.width()  - 1)))
#         ry = max(0.0, min((lp.y() - oy) / scale, float(pm.height() - 1)))
#         return int(round(rx)), int(round(ry))

#     # ======================================================================= #
#     #  TEMPLATE SELECT                                                         #
#     # ======================================================================= #
#     def _update_select_buttons_state(self):
#         sel = self._select_mode is not None
#         self.btn_select_template.setEnabled(not sel)
#         self.btn_cancle_select.setEnabled(sel)

#     def _on_select_template(self):
#         if self._current_image_path is None:
#             QtWidgets.QMessageBox.warning(self, "Chưa chọn ảnh",
#                                           "Vui lòng chọn ảnh từ danh sách trước.")
#             return
#         self._select_mode      = "waiting_first"
#         self._template_point_1 = None
#         self._template_point_2 = None
#         self._update_select_buttons_state()
#         self._display_image(self._current_image_path)

#     def _on_cancel_select(self):
#         self._select_mode      = None
#         self._template_point_1 = None
#         self._template_point_2 = None
#         self._update_select_buttons_state()
#         if self._current_image_path:
#             self._display_image(self._current_image_path)

#     def _on_image_clicked(self, point: QPoint):
#         if self._select_mode is None:
#             return
#         if self._select_mode == "waiting_first":
#             self._template_point_1 = point
#             self._select_mode      = "waiting_second"
#             self._update_select_buttons_state()
#             self._display_image(self._current_image_path, point1=point)
#         elif self._select_mode == "waiting_second":
#             self._template_point_2 = point
#             self._display_image(self._current_image_path,
#                                 point1=self._template_point_1, point2=point)
#             self._save_template_point(self._template_point_1, point)
#             self._select_mode      = None
#             self._template_point_1 = None
#             self._template_point_2 = None
#             self._update_select_buttons_state()

#     def _save_template_point(self, p1: QPoint, p2: QPoint):
#         json_path = self._get_template_point_path()
#         if json_path is None:
#             return
#         x1, y1 = self._label_to_img(p1)
#         x2, y2 = self._label_to_img(p2)
#         rx, ry  = min(x1, x2), min(y1, y2)
#         rw, rh  = abs(x2 - x1), abs(y2 - y1)
#         data = {"image_path": self._current_image_path,
#                 "point1": {"x": x1, "y": y1}, "point2": {"x": x2, "y": y2},
#                 "rect":   {"x": rx, "y": ry,  "w": rw,  "h": rh}}
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         QtWidgets.QMessageBox.information(
#             self, "Lưu thành công",
#             f"Rect: x={rx}, y={ry}, w={rw}, h={rh}"
#         )

#     # ======================================================================= #
#     #  LIQUID LEVEL SETUP                                                      #
#     # ======================================================================= #
#     def _on_setup_liquid(self):
#         """
#         1. Validate ảnh + rect
#         2. Crop template gray
#         3. Khởi động LiquidSetupWorker → U2Net (instance riêng) + detect
#         4. detect_done → _on_liquid_detect_done → hỏi tolerance → save_final
#         """
#         if self._current_image_path is None:
#             QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh.")
#             return

#         json_path = self._get_template_point_path()
#         if json_path is None or not json_path.exists():
#             QtWidgets.QMessageBox.warning(self, "Chưa chọn vùng ROI",
#                                           "Dùng 'Select Template' để khoanh vùng mặt nước trước.")
#             return

#         with open(json_path, "r", encoding="utf-8") as f:
#             tp = json.load(f)
#         rect = tp.get("rect", {})
#         if not rect or rect.get("w", 0) <= 0 or rect.get("h", 0) <= 0:
#             QtWidgets.QMessageBox.warning(self, "Lỗi", "Vùng chọn không hợp lệ.")
#             return

#         x, y, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
#         img = cv2.imread(self._current_image_path)
#         if img is None:
#             QtWidgets.QMessageBox.warning(self, "Lỗi",
#                                           f"Không đọc được ảnh:\n{self._current_image_path}")
#             return

#         gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         template = gray[y: y + rh, x: x + rw]

#         pr = self._get_project_root()
#         if pr is None:
#             return

#         self.btn_setup_liquid.setEnabled(False)

#         self._liquid_worker = LiquidSetupWorker(
#             img_bgr       = img,
#             template_gray = template,
#             x=x, w=rw, h=rh,
#             project_root  = pr,
#         )
#         self._liquid_worker.detect_done.connect(self._on_liquid_detect_done)
#         self._liquid_worker.start()

#     def _on_liquid_detect_done(self, success: bool, fill_ratio: float,
#                                display_img, error_msg: str):
#         """Callback trên UI thread sau khi worker detect xong."""
#         self.btn_setup_liquid.setEnabled(True)

#         if not success:
#             QtWidgets.QMessageBox.warning(self, "Detect thất bại", error_msg)
#             self._liquid_worker = None
#             return

#         if display_img is not None:
#             self._display_cv2_image(display_img)

#         tolerance, ok = QtWidgets.QInputDialog.getDouble(
#             self, "Dung sai mực nước",
#             f"Fill ratio đo được: {fill_ratio:.1f}%\n\nNhập dung sai ± (%)",
#             value=10.0, min=0.5, max=50.0, decimals=1
#         )
#         if not ok:
#             self._liquid_worker = None
#             return

#         # Lưu config + hiển thị preview cuối
#         final_img = self._liquid_worker.save_final(tolerance)
#         if final_img is not None:
#             self._display_cv2_image(final_img)

#         min_f = max(0.0,   fill_ratio - tolerance)
#         max_f = min(100.0, fill_ratio + tolerance)
#         QtWidgets.QMessageBox.information(
#             self, "Setup hoàn tất",
#             f"Fill ratio  : {fill_ratio:.1f}%\n"
#             f"Dung sai    : ± {tolerance:.1f}%\n"
#             f"Ngưỡng OK   : [{min_f:.1f}% – {max_f:.1f}%]\n\n"
#             f"(U2Net mask đã được dùng để detect mực nước)"
#         )
#         self._liquid_worker = None

#     # ======================================================================= #
#     #  CAMERA                                                                  #
#     # ======================================================================= #
#     def init_camera(self):
#         if self.is_initialized:
#             return
#         try:
#             self.data_manager = GetDataManager()
#             self.data_manager.start()
#             self.is_initialized = True
#         except Exception as e:
#             print(f"[GetData] ❌ Khởi tạo thất bại: {e}")

#     def start_get_data(self):
#         if self.is_saving:
#             return
#         save_dir = self._get_goods_path()
#         if not save_dir:
#             return
#         self.init_camera()
#         if not self.is_initialized:
#             return
#         self._set_tree_root(str(save_dir))
#         self.data_manager.set_save_dir(str(save_dir))
#         self.data_manager.is_saving = True
#         self.is_saving = True
#         print(f"[GetData] ▶ Thu thập → {save_dir}")

#     def stop_get_data(self):
#         if not self.is_saving:
#             return
#         self.data_manager.is_saving = False
#         self.is_saving = False
#         print("[GetData] ⏹ Dừng thu thập.")

#     # ======================================================================= #
#     #  BUILD MEMORY BANK                                                       #
#     # ======================================================================= #
#     def build_memory(self):
#         if self._build_worker and self._build_worker.isRunning():
#             return
#         goods_dir  = self._get_goods_path()
#         output_dir = self._get_memory_bank_dir()
#         if goods_dir is None or output_dir is None:
#             return
#         self.stop_get_data()
#         self._set_data_buttons_enabled(False)
#         self._build_worker = BuildBankWorker(goods_dir, output_dir)
#         self._build_worker.finished.connect(self._on_build_finished)
#         self._build_worker.start()

#     def _on_build_finished(self, success: bool, message: str):
#         print(f"[GetData] {message}")
#         self._set_data_buttons_enabled(True)

#     # ======================================================================= #
#     #  MISC                                                                    #
#     # ======================================================================= #
#     def refresh_projects(self):
#         current = self.name_project.currentText()
#         self._populate_project_combo()
#         idx = self.name_project.findText(current)
#         if idx >= 0:
#             self.name_project.setCurrentIndex(idx)
#         self._update_tree_from_combo()

#     def closeEvent(self, event):
#         if self._build_worker and self._build_worker.isRunning():
#             self._build_worker.quit()
#             self._build_worker.wait(3000)
#         if self._liquid_worker and self._liquid_worker.isRunning():
#             self._liquid_worker.quit()
#             self._liquid_worker.wait(3000)
#         if self.data_manager:
#             self.data_manager.stop()
#             self.data_manager.join(timeout=3)
#         super().closeEvent(event)


# # =========================================================================== #
# #  GET DATA MANAGER                                                            #
# # =========================================================================== #
# class GetDataManager(threading.Thread):
#     def __init__(self):
#         super().__init__(daemon=True, name="GetDataManager")
#         self.frames_queue   = queue.Queue(maxsize=1)
#         self._stop_event    = threading.Event()
#         self.is_saving      = False
#         self.save_dir       = None
#         self._trigger_count = 0

#     def set_save_dir(self, path: str):
#         self.save_dir       = path
#         self._trigger_count = self._scan_last_index(path)

#     @staticmethod
#     def _scan_last_index(directory: str) -> int:
#         if not os.path.isdir(directory):
#             return 0
#         pattern   = re.compile(r"^cam\d+_(\d+)\.jpg$", re.IGNORECASE)
#         max_index = 0
#         for fname in os.listdir(directory):
#             m = pattern.match(fname)
#             if m:
#                 max_index = max(max_index, int(m.group(1)))
#         return max_index

#     def stop(self):
#         self._stop_event.set()

#     def run(self):
#         thread_camera = None
#         try:
#             json_path = Path(PROJECTS_ROOT)   # fallback
#             infor     = None
#             thread_camera = BatchCamera(self.frames_queue, self._stop_event, infor)
#             thread_camera.start()
#             while not self._stop_event.is_set():
#                 try:
#                     _, frames = self.frames_queue.get(timeout=1)
#                 except queue.Empty:
#                     continue
#                 if self.is_saving and self.save_dir:
#                     self._save_trigger(frames)
#         except Exception as e:
#             print(f"[GetDataManager] ❌ {e}")
#         finally:
#             if thread_camera:
#                 thread_camera.join(timeout=3)

#     def _save_trigger(self, frames: list):
#         self._trigger_count += 1
#         for cam_id, frame in enumerate(frames):
#             if frame is None:
#                 continue
#             fname = f"cam{cam_id+1}_{self._trigger_count:04d}.jpg"
#             path  = os.path.join(self.save_dir, fname)
#             cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#         print(f"[GetDataManager] 💾 trigger_{self._trigger_count:04d}")






# from PyQt6 import uic, QtWidgets
# from PyQt6.QtCore import QThread, pyqtSignal, QDir, QPoint, QRect, Qt
# from PyQt6.QtGui import QFileSystemModel, QPixmap, QPainter, QPen, QColor, QBrush, QImage
# import queue
# from core.path_manager import BASE_DIR
# import os
# import re
# import threading
# import cv2
# import numpy as np
# from pathlib import Path
# from hardware.camera.batch_camera import BatchCamera
# from training.patchcore_memory_bank import build_bank
# import json


# PROJECTS_ROOT           = os.path.join(BASE_DIR, "projects")
# PROJECT_INFO_FILENAME   = "project_info.json"
# TEMPLATE_POINT_FILENAME = "template_point.json"
# U2NET_ENGINE_PATH       = os.path.join(BASE_DIR, "models/remove_bg/u2netp.trt")


# # =========================================================================== #
# #  BUILD WORKER                                                                #
# # =========================================================================== #
# class BuildBankWorker(QThread):
#     finished = pyqtSignal(bool, str)

#     def __init__(self, goods_dir: Path, output_dir: Path):
#         super().__init__()
#         self.goods_dir  = goods_dir
#         self.output_dir = output_dir

#     def run(self):
#         try:
#             build_bank(self.goods_dir, self.output_dir)
#             self.finished.emit(True, f"✅ Build hoàn tất.\n{self.output_dir / 'memory_bank.pt'}")
#         except Exception as e:
#             self.finished.emit(False, f"❌ Build thất bại: {e}")


# # =========================================================================== #
# #  LIQUID SETUP WORKER                                                         #
# #  Tạo U2NetSegmentor riêng (cùng code như BodyWorker, dùng chung engine file) #
# # =========================================================================== #
# class LiquidSetupWorker(QThread):
#     """
#     Chạy trong QThread riêng để không block UI:
#       1. Tạo U2NetSegmentor (cùng engine file với BodyWorker)
#       2. attach() → get_mask() → detach()
#       3. LiquidLevelDetector.detect(img, mask) → baseline fill_ratio
#       4. emit detect_done

#     Sau khi user nhập tolerance → gọi save_final() (từ UI thread, an toàn).
#     """
#     # (success, fill_ratio, display_img_bgr, error_msg)
#     detect_done = pyqtSignal(bool, float, object, str)

#     def __init__(self, img_bgr: np.ndarray,
#                  template_gray: np.ndarray,
#                  x: int, w: int, h: int,
#                  project_root: Path):
#         super().__init__()
#         self.img_bgr       = img_bgr
#         self.template_gray = template_gray
#         self.x             = x
#         self.w             = w
#         self.h             = h
#         self.project_root  = project_root

#         # Được set sau khi detect xong, dùng trong save_final()
#         self._baseline    = None
#         self._detector    = None
#         self._object_mask = None

#     # ------------------------------------------------------------------ #
#     def run(self):
#         from core.liquid_level import LiquidLevelDetector

#         detector = LiquidLevelDetector(self.project_root)

#         # Lưu config tạm để detect() có thể chạy
#         detector.save_setup(
#             self.template_gray,
#             x=self.x, w=self.w, h=self.h,
#             baseline=50.0, tolerance=10.0,
#         )

#         # ── Lấy mask từ U2Net (tương tự cách BodyWorker làm) ─────────────
#         object_mask = None
#         if os.path.exists(U2NET_ENGINE_PATH):
#             try:
#                 from core.u2net_segmentor import U2NetSegmentor
#                 seg = U2NetSegmentor(U2NET_ENGINE_PATH)   # instance riêng, cùng engine
#                 seg.attach()
#                 object_mask = seg.get_mask(self.img_bgr)
#                 seg.detach()
#                 print("[LiquidSetup] ✅ U2Net mask OK.")
#             except Exception as e:
#                 print(f"[LiquidSetup] ⚠️  U2Net thất bại: {e} — detect không có mask")
#         else:
#             print(f"[LiquidSetup] ⚠️  Không tìm thấy engine — detect không có mask")

#         # ── Detect → lấy baseline ─────────────────────────────────────────
#         try:
#             result = detector.detect(self.img_bgr, object_mask=object_mask)
#         except Exception as e:
#             self.detect_done.emit(False, 0.0, None, f"detect() lỗi: {e}")
#             return

#         if result is None:
#             self.detect_done.emit(
#                 False, 0.0, None,
#                 "Không tìm được mực nước.\nHãy chọn lại vùng ROI chính xác hơn."
#             )
#             return

#         # Lưu lại để save_final() dùng
#         self._baseline    = result["fill_ratio"]
#         self._detector    = detector
#         self._object_mask = object_mask

#         self.detect_done.emit(True, self._baseline, result["display_img"], "")

#     # ------------------------------------------------------------------ #
#     def save_final(self, tolerance: float) -> np.ndarray | None:
#         """
#         Gọi từ UI thread sau khi user nhập tolerance.
#         Lưu config chính thức + chạy detect lần cuối để có display_img đúng.
#         Trả về display_img (BGR numpy) hoặc None.
#         """
#         if self._detector is None or self._baseline is None:
#             return None

#         self._detector.save_setup(
#             self.template_gray,
#             x=self.x, w=self.w, h=self.h,
#             baseline=self._baseline,
#             tolerance=tolerance,
#         )
#         result = self._detector.detect(self.img_bgr, object_mask=self._object_mask)
#         return result["display_img"] if result else None


# # =========================================================================== #
# #  GET DATA TAB                                                                #
# # =========================================================================== #
# class GetDataTab(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         ui_path = os.path.join(BASE_DIR, "ui", "ui", "get_data_tab.ui")
#         uic.loadUi(ui_path, self)

#         self.btn_start_get_data.clicked.connect(self.start_get_data)
#         self.btn_stop_get_data.clicked.connect(self.stop_get_data)
#         self.build_memory_bank.clicked.connect(self.build_memory)
#         self.btn_select_template.clicked.connect(self._on_select_template)
#         self.btn_cancle_select.clicked.connect(self._on_cancel_select)
#         self.btn_setup_liquid.clicked.connect(self._on_setup_liquid)

#         self.data_manager        = None
#         self.is_initialized      = False
#         self.is_saving           = False
#         self._build_worker       = None
#         self._liquid_worker: LiquidSetupWorker | None = None

#         self._select_mode        = None
#         self._template_point_1   = None
#         self._template_point_2   = None
#         self._current_image_path = None

#         self.img_show.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.img_show.clicked.connect(self._on_image_clicked)

#         self._fs_model = QFileSystemModel()
#         self._fs_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
#         self.treeView_img.setModel(self._fs_model)
#         self.treeView_img.setColumnHidden(1, True)
#         self.treeView_img.setColumnHidden(2, True)
#         self.treeView_img.setColumnHidden(3, True)
#         self.treeView_img.setHeaderHidden(False)
#         self.treeView_img.selectionModel().selectionChanged.connect(
#             self._on_tree_selection_changed
#         )

#         self._populate_project_combo()
#         self.name_project.currentIndexChanged.connect(self._on_project_changed)
#         self._update_tree_from_combo()
#         self._update_select_buttons_state()

#     # ======================================================================= #
#     #  PROJECT HELPERS                                                         #
#     # ======================================================================= #
#     def _get_projects_root(self) -> str:
#         return PROJECTS_ROOT

#     def _populate_project_combo(self):
#         self.name_project.blockSignals(True)
#         self.name_project.clear()
#         root = self._get_projects_root()
#         if os.path.isdir(root):
#             entries = sorted(e for e in os.listdir(root)
#                              if os.path.isdir(os.path.join(root, e)))
#             self.name_project.addItems(entries)
#         self.name_project.blockSignals(False)

#     def _on_project_changed(self, _):
#         self._update_tree_from_combo()

#     def _update_tree_from_combo(self):
#         goods = self._get_goods_path()
#         if goods:
#             self._set_tree_root(str(goods))

#     def _get_project_root(self) -> Path | None:
#         name = self.name_project.currentText().strip()
#         return Path(self._get_projects_root()) / name if name else None

#     def _get_goods_path(self) -> Path | None:
#         pr = self._get_project_root()
#         if pr is None:
#             return None
#         g = pr / "goods"
#         g.mkdir(parents=True, exist_ok=True)
#         return g

#     def _get_memory_bank_dir(self) -> Path | None:
#         pr = self._get_project_root()
#         if pr is None:
#             return None
#         d = pr / "memory_bank"
#         d.mkdir(parents=True, exist_ok=True)
#         return d

#     def _get_template_point_path(self) -> Path | None:
#         pr = self._get_project_root()
#         return pr / TEMPLATE_POINT_FILENAME if pr else None

#     # ======================================================================= #
#     #  TREE VIEW                                                               #
#     # ======================================================================= #
#     def _set_tree_root(self, path: str):
#         idx = self._fs_model.setRootPath(path)
#         self.treeView_img.setRootIndex(idx)
#         self.treeView_img.expandAll()

#     def _set_data_buttons_enabled(self, enabled: bool):
#         self.btn_start_get_data.setEnabled(enabled)
#         self.btn_stop_get_data.setEnabled(enabled)
#         self.build_memory_bank.setEnabled(enabled)

#     def _on_tree_selection_changed(self, selected, _):
#         indexes = selected.indexes()
#         if not indexes:
#             return
#         file_path = self._fs_model.filePath(indexes[0])
#         if not os.path.isfile(file_path):
#             return
#         if os.path.splitext(file_path)[1].lower() not in \
#                 (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
#             return
#         self._current_image_path = file_path
#         self._select_mode        = None
#         self._template_point_1   = None
#         self._template_point_2   = None
#         self._update_select_buttons_state()
#         self._display_image(file_path)

#     # ======================================================================= #
#     #  DISPLAY HELPERS                                                         #
#     # ======================================================================= #
#     def _get_scale_offset(self, orig_w, orig_h):
#         lw, lh   = self.img_show.width(), self.img_show.height()
#         scale    = min(lw / orig_w, lh / orig_h)
#         sw, sh   = int(orig_w * scale), int(orig_h * scale)
#         ox, oy   = (lw - sw) // 2, (lh - sh) // 2
#         return scale, sw, sh, ox, oy

#     def _display_image(self, file_path: str,
#                        point1: QPoint | None = None,
#                        point2: QPoint | None = None):
#         pm = QPixmap(file_path)
#         if pm.isNull():
#             return
#         ow, oh = pm.width(), pm.height()
#         scale, sw, sh, ox, oy = self._get_scale_offset(ow, oh)
#         spm = pm.scaled(sw, sh,
#                         Qt.AspectRatioMode.KeepAspectRatio,
#                         Qt.TransformationMode.SmoothTransformation)
#         if point1 is not None:
#             px1, py1 = int(point1.x() - ox), int(point1.y() - oy)
#             p = QPainter(spm)
#             p.setRenderHint(QPainter.RenderHint.Antialiasing)
#             if point2 is None:
#                 p.setPen(QPen(QColor(0, 255, 0), 2))
#                 p.setBrush(Qt.BrushStyle.NoBrush)
#                 p.drawEllipse(QPoint(px1, py1), 6, 6)
#                 p.drawLine(px1 - 10, py1, px1 + 10, py1)
#                 p.drawLine(px1, py1 - 10, px1, py1 + 10)
#             else:
#                 px2, py2 = int(point2.x() - ox), int(point2.y() - oy)
#                 rect = QRect(QPoint(px1, py1), QPoint(px2, py2)).normalized()
#                 p.setBrush(QBrush(QColor(0, 255, 0, 40)))
#                 p.setPen(QPen(QColor(0, 255, 0), 2))
#                 p.drawRect(rect)
#                 p.setBrush(QBrush(QColor(0, 255, 0)))
#                 p.setPen(Qt.PenStyle.NoPen)
#                 p.drawEllipse(QPoint(px1, py1), 4, 4)
#                 p.drawEllipse(QPoint(px2, py2), 4, 4)
#             p.end()
#         self.img_show.setPixmap(spm)

#     def _display_cv2_image(self, img_bgr: np.ndarray):
#         rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
#         pm    = QPixmap.fromImage(qimg)
#         scaled = pm.scaled(self.img_show.width(), self.img_show.height(),
#                            Qt.AspectRatioMode.KeepAspectRatio,
#                            Qt.TransformationMode.SmoothTransformation)
#         self.img_show.setPixmap(scaled)

#     # ======================================================================= #
#     #  COORDINATE CONVERSION                                                   #
#     # ======================================================================= #
#     def _label_to_img(self, lp: QPoint) -> tuple[int, int]:
#         if not self._current_image_path:
#             return lp.x(), lp.y()
#         pm = QPixmap(self._current_image_path)
#         if pm.isNull():
#             return lp.x(), lp.y()
#         scale, _, _, ox, oy = self._get_scale_offset(pm.width(), pm.height())
#         rx = max(0.0, min((lp.x() - ox) / scale, float(pm.width()  - 1)))
#         ry = max(0.0, min((lp.y() - oy) / scale, float(pm.height() - 1)))
#         return int(round(rx)), int(round(ry))

#     # ======================================================================= #
#     #  TEMPLATE SELECT                                                         #
#     # ======================================================================= #
#     def _update_select_buttons_state(self):
#         sel = self._select_mode is not None
#         self.btn_select_template.setEnabled(not sel)
#         self.btn_cancle_select.setEnabled(sel)

#     def _on_select_template(self):
#         if self._current_image_path is None:
#             QtWidgets.QMessageBox.warning(self, "Chưa chọn ảnh",
#                                           "Vui lòng chọn ảnh từ danh sách trước.")
#             return
#         self._select_mode      = "waiting_first"
#         self._template_point_1 = None
#         self._template_point_2 = None
#         self._update_select_buttons_state()
#         self._display_image(self._current_image_path)

#     def _on_cancel_select(self):
#         self._select_mode      = None
#         self._template_point_1 = None
#         self._template_point_2 = None
#         self._update_select_buttons_state()
#         if self._current_image_path:
#             self._display_image(self._current_image_path)

#     def _on_image_clicked(self, point: QPoint):
#         if self._select_mode is None:
#             return
#         if self._select_mode == "waiting_first":
#             self._template_point_1 = point
#             self._select_mode      = "waiting_second"
#             self._update_select_buttons_state()
#             self._display_image(self._current_image_path, point1=point)
#         elif self._select_mode == "waiting_second":
#             self._template_point_2 = point
#             self._display_image(self._current_image_path,
#                                 point1=self._template_point_1, point2=point)
#             self._save_template_point(self._template_point_1, point)
#             self._select_mode      = None
#             self._template_point_1 = None
#             self._template_point_2 = None
#             self._update_select_buttons_state()

#     def _save_template_point(self, p1: QPoint, p2: QPoint):
#         json_path = self._get_template_point_path()
#         if json_path is None:
#             return
#         x1, y1 = self._label_to_img(p1)
#         x2, y2 = self._label_to_img(p2)
#         rx, ry  = min(x1, x2), min(y1, y2)
#         rw, rh  = abs(x2 - x1), abs(y2 - y1)
#         data = {"image_path": self._current_image_path,
#                 "point1": {"x": x1, "y": y1}, "point2": {"x": x2, "y": y2},
#                 "rect":   {"x": rx, "y": ry,  "w": rw,  "h": rh}}
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         QtWidgets.QMessageBox.information(
#             self, "Lưu thành công",
#             f"Rect: x={rx}, y={ry}, w={rw}, h={rh}"
#         )

#     # ======================================================================= #
#     #  LIQUID LEVEL SETUP                                                      #
#     # ======================================================================= #
#     def _on_setup_liquid(self):
#         """
#         1. Validate ảnh + rect
#         2. Crop template gray
#         3. Khởi động LiquidSetupWorker → U2Net (instance riêng) + detect
#         4. detect_done → _on_liquid_detect_done → hỏi tolerance → save_final
#         """
#         if self._current_image_path is None:
#             QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh.")
#             return

#         json_path = self._get_template_point_path()
#         if json_path is None or not json_path.exists():
#             QtWidgets.QMessageBox.warning(self, "Chưa chọn vùng ROI",
#                                           "Dùng 'Select Template' để khoanh vùng mặt nước trước.")
#             return

#         with open(json_path, "r", encoding="utf-8") as f:
#             tp = json.load(f)
#         rect = tp.get("rect", {})
#         if not rect or rect.get("w", 0) <= 0 or rect.get("h", 0) <= 0:
#             QtWidgets.QMessageBox.warning(self, "Lỗi", "Vùng chọn không hợp lệ.")
#             return

#         x, y, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
#         img = cv2.imread(self._current_image_path)
#         if img is None:
#             QtWidgets.QMessageBox.warning(self, "Lỗi",
#                                           f"Không đọc được ảnh:\n{self._current_image_path}")
#             return

#         gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         template = gray[y: y + rh, x: x + rw]

#         pr = self._get_project_root()
#         if pr is None:
#             return

#         self.btn_setup_liquid.setEnabled(False)

#         self._liquid_worker = LiquidSetupWorker(
#             img_bgr       = img,
#             template_gray = template,
#             x=x, w=rw, h=rh,
#             project_root  = pr,
#         )
#         self._liquid_worker.detect_done.connect(self._on_liquid_detect_done)
#         self._liquid_worker.start()

#     def _on_liquid_detect_done(self, success: bool, fill_ratio: float,
#                                display_img, error_msg: str):
#         """Callback trên UI thread sau khi worker detect xong."""
#         self.btn_setup_liquid.setEnabled(True)

#         if not success:
#             QtWidgets.QMessageBox.warning(self, "Detect thất bại", error_msg)
#             self._liquid_worker = None
#             return

#         if display_img is not None:
#             self._display_cv2_image(display_img)

#         tolerance, ok = QtWidgets.QInputDialog.getDouble(
#             self, "Dung sai mực nước",
#             f"Fill ratio đo được: {fill_ratio:.1f}%\n\nNhập dung sai ± (%)",
#             value=10.0, min=0.5, max=50.0, decimals=1
#         )
#         if not ok:
#             self._liquid_worker = None
#             return

#         # Lưu config + hiển thị preview cuối
#         final_img = self._liquid_worker.save_final(tolerance)
#         if final_img is not None:
#             self._display_cv2_image(final_img)

#         min_f = max(0.0,   fill_ratio - tolerance)
#         max_f = min(100.0, fill_ratio + tolerance)
#         QtWidgets.QMessageBox.information(
#             self, "Setup hoàn tất",
#             f"Fill ratio  : {fill_ratio:.1f}%\n"
#             f"Dung sai    : ± {tolerance:.1f}%\n"
#             f"Ngưỡng OK   : [{min_f:.1f}% – {max_f:.1f}%]\n\n"
#             f"(U2Net mask đã được dùng để detect mực nước)"
#         )
#         self._liquid_worker = None

#     # ======================================================================= #
#     #  CAMERA                                                                  #
#     # ======================================================================= #
#     def init_camera(self):
#         if self.is_initialized:
#             return
#         pr = self._get_project_root()
#         if pr is None:
#             print("[GetData] ⚠️  Chưa chọn project.")
#             return
#         try:
#             self.data_manager = GetDataManager(project_root=pr)
#             self.data_manager.start()
#             self.is_initialized = True
#             print(f"[GetData] ✅ Camera khởi tạo — project: {pr.name}")
#         except Exception as e:
#             print(f"[GetData] ❌ Khởi tạo thất bại: {e}")

#     def start_get_data(self):
#         if self.is_saving:
#             return
#         save_dir = self._get_goods_path()
#         if not save_dir:
#             return
#         self.init_camera()
#         if not self.is_initialized:
#             return
#         self._set_tree_root(str(save_dir))
#         self.data_manager.set_save_dir(str(save_dir))
#         self.data_manager.is_saving = True
#         self.is_saving = True
#         print(f"[GetData] ▶ Thu thập → {save_dir}")

#     def stop_get_data(self):
#         if not self.is_saving:
#             return
#         self.data_manager.is_saving = False
#         self.is_saving = False
#         print("[GetData] ⏹ Dừng thu thập.")

#     # ======================================================================= #
#     #  BUILD MEMORY BANK                                                       #
#     # ======================================================================= #
#     def build_memory(self):
#         if self._build_worker and self._build_worker.isRunning():
#             return
#         goods_dir  = self._get_goods_path()
#         output_dir = self._get_memory_bank_dir()
#         if goods_dir is None or output_dir is None:
#             return
#         self.stop_get_data()
#         self._set_data_buttons_enabled(False)
#         self._build_worker = BuildBankWorker(goods_dir, output_dir)
#         self._build_worker.finished.connect(self._on_build_finished)
#         self._build_worker.start()

#     def _on_build_finished(self, success: bool, message: str):
#         print(f"[GetData] {message}")
#         self._set_data_buttons_enabled(True)

#     # ======================================================================= #
#     #  MISC                                                                    #
#     # ======================================================================= #
#     def refresh_projects(self):
#         current = self.name_project.currentText()
#         self._populate_project_combo()
#         idx = self.name_project.findText(current)
#         if idx >= 0:
#             self.name_project.setCurrentIndex(idx)
#         self._update_tree_from_combo()

#     def closeEvent(self, event):
#         if self._build_worker and self._build_worker.isRunning():
#             self._build_worker.quit()
#             self._build_worker.wait(3000)
#         if self._liquid_worker and self._liquid_worker.isRunning():
#             self._liquid_worker.quit()
#             self._liquid_worker.wait(3000)
#         if self.data_manager:
#             self.data_manager.stop()
#             self.data_manager.join(timeout=3)
#         super().closeEvent(event)


# # =========================================================================== #
# #  GET DATA MANAGER                                                            #
# # =========================================================================== #
# class GetDataManager(threading.Thread):
#     def __init__(self, project_root: Path):
#         super().__init__(daemon=True, name="GetDataManager")
#         self.project_root   = project_root
#         self.frames_queue   = queue.Queue(maxsize=1)
#         self._stop_event    = threading.Event()
#         self.is_saving      = False
#         self.save_dir       = None
#         self._trigger_count = 0

#     def set_save_dir(self, path: str):
#         self.save_dir       = path
#         self._trigger_count = self._scan_last_index(path)
#         if self._trigger_count > 0:
#             print(f"[GetDataManager] 🔄 Resume từ trigger_{self._trigger_count:04d}")
#         else:
#             print("[GetDataManager] 🆕 Bắt đầu từ 0001")

#     def _load_project_info(self) -> dict | None:
#         json_path = self.project_root / PROJECT_INFO_FILENAME
#         if not json_path.exists():
#             print(f"[GetDataManager] ⚠️  Không tìm thấy {json_path}")
#             return None
#         with open(json_path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     @staticmethod
#     def _scan_last_index(directory: str) -> int:
#         if not os.path.isdir(directory):
#             return 0
#         pattern   = re.compile(r"^cam\d+_(\d+)\.jpg$", re.IGNORECASE)
#         max_index = 0
#         for fname in os.listdir(directory):
#             m = pattern.match(fname)
#             if m:
#                 max_index = max(max_index, int(m.group(1)))
#         return max_index

#     def stop(self):
#         self._stop_event.set()

#     def run(self):
#         thread_camera = None
#         try:
#             infor_project = self._load_project_info()
#             thread_camera = BatchCamera(self.frames_queue, self._stop_event, infor_project)
#             thread_camera.start()
#             print("[GetDataManager] Camera started.")
#             while not self._stop_event.is_set():
#                 try:
#                     _, frames = self.frames_queue.get(timeout=1)
#                 except queue.Empty:
#                     continue
#                 if self.is_saving and self.save_dir:
#                     self._save_trigger(frames)
#         except Exception as e:
#             print(f"[GetDataManager] ❌ {e}")
#         finally:
#             if thread_camera:
#                 thread_camera.join(timeout=3)
#             print("[GetDataManager] Stopped.")

#     def _save_trigger(self, frames: list):
#         self._trigger_count += 1
#         for cam_id, frame in enumerate(frames):
#             if frame is None:
#                 continue
#             fname = f"cam{cam_id+1}_{self._trigger_count:04d}.jpg"
#             path  = os.path.join(self.save_dir, fname)
#             cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#         print(f"[GetDataManager] 💾 trigger_{self._trigger_count:04d}")



from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal, QDir, QPoint, QRect, Qt
from PyQt6.QtGui import QFileSystemModel, QPixmap, QPainter, QPen, QColor, QBrush, QImage
import queue
from core.path_manager import BASE_DIR
import os
import re
import threading
import cv2
import numpy as np
from pathlib import Path
from hardware.camera.batch_camera import BatchCamera
from training.patchcore_memory_bank import build_bank
import json


PROJECTS_ROOT           = os.path.join(BASE_DIR, "projects")
PROJECT_INFO_FILENAME   = "project_info.json"
TEMPLATE_POINT_FILENAME = "template_point.json"
U2NET_ENGINE_PATH       = os.path.join(BASE_DIR, "models/remove_bg/u2netp.trt")


# =========================================================================== #
#  BUILD WORKER                                                                #
# =========================================================================== #
class BuildBankWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, goods_dir: Path, output_dir: Path):
        super().__init__()
        self.goods_dir  = goods_dir
        self.output_dir = output_dir

    def run(self):
        try:
            build_bank(self.goods_dir, self.output_dir)
            self.finished.emit(True, f"✅ Build hoàn tất.\n{self.output_dir / 'memory_bank.pt'}")
        except Exception as e:
            self.finished.emit(False, f"❌ Build thất bại: {e}")


# =========================================================================== #
#  LIQUID SETUP WORKER                                                         #
# =========================================================================== #
class LiquidSetupWorker(QThread):
    detect_done = pyqtSignal(bool, float, object, str)

    def __init__(self, img_bgr: np.ndarray,
                 template_gray: np.ndarray,
                 x: int, w: int, h: int,
                 project_root: Path):
        super().__init__()
        self.img_bgr       = img_bgr
        self.template_gray = template_gray
        self.x             = x
        self.w             = w
        self.h             = h
        self.project_root  = project_root

        self._baseline    = None
        self._detector    = None
        self._object_mask = None

    def run(self):
        from core.liquid_level import LiquidLevelDetector

        detector = LiquidLevelDetector(self.project_root)
        detector.save_setup(
            self.template_gray,
            x=self.x, w=self.w, h=self.h,
            baseline=50.0, tolerance=10.0,
        )

        object_mask = None
        if os.path.exists(U2NET_ENGINE_PATH):
            try:
                from core.u2net_segmentor import U2NetSegmentor
                seg = U2NetSegmentor(U2NET_ENGINE_PATH)
                seg.attach()
                object_mask = seg.get_mask(self.img_bgr)
                seg.detach()
                print("[LiquidSetup] ✅ U2Net mask OK.")
            except Exception as e:
                print(f"[LiquidSetup] ⚠️  U2Net thất bại: {e} — detect không có mask")
        else:
            print(f"[LiquidSetup] ⚠️  Không tìm thấy engine — detect không có mask")

        try:
            result = detector.detect(self.img_bgr, object_mask=object_mask)
        except Exception as e:
            self.detect_done.emit(False, 0.0, None, f"detect() lỗi: {e}")
            return

        if result is None:
            self.detect_done.emit(
                False, 0.0, None,
                "Không tìm được mực nước.\nHãy chọn lại vùng ROI chính xác hơn."
            )
            return

        self._baseline    = result["fill_ratio"]
        self._detector    = detector
        self._object_mask = object_mask

        self.detect_done.emit(True, self._baseline, result["display_img"], "")

    def save_final(self, tolerance: float) -> np.ndarray | None:
        if self._detector is None or self._baseline is None:
            return None

        self._detector.save_setup(
            self.template_gray,
            x=self.x, w=self.w, h=self.h,
            baseline=self._baseline,
            tolerance=tolerance,
        )
        result = self._detector.detect(self.img_bgr, object_mask=self._object_mask)
        return result["display_img"] if result else None


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
        self.btn_select_template.clicked.connect(self._on_select_template)
        self.btn_cancle_select.clicked.connect(self._on_cancel_select)
        self.btn_setup_liquid.clicked.connect(self._on_setup_liquid)

        self.data_manager        = None
        self.is_initialized      = False
        self.is_saving           = False
        self._build_worker       = None
        self._liquid_worker: LiquidSetupWorker | None = None

        self._select_mode        = None
        self._template_point_1   = None
        self._template_point_2   = None
        self._current_image_path = None

        self.img_show.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_show.clicked.connect(self._on_image_clicked)

        self._fs_model = QFileSystemModel()
        self._fs_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
        self.treeView_img.setModel(self._fs_model)
        self.treeView_img.setColumnHidden(1, True)
        self.treeView_img.setColumnHidden(2, True)
        self.treeView_img.setColumnHidden(3, True)
        self.treeView_img.setHeaderHidden(False)
        self.treeView_img.selectionModel().selectionChanged.connect(
            self._on_tree_selection_changed
        )

        self._populate_project_combo()
        self.name_project.currentIndexChanged.connect(self._on_project_changed)
        self._update_tree_from_combo()
        self._update_select_buttons_state()

    # ======================================================================= #
    #  PROJECT HELPERS                                                         #
    # ======================================================================= #
    def _get_projects_root(self) -> str:
        return PROJECTS_ROOT

    def _populate_project_combo(self):
        self.name_project.blockSignals(True)
        self.name_project.clear()
        root = self._get_projects_root()
        if os.path.isdir(root):
            entries = sorted(e for e in os.listdir(root)
                             if os.path.isdir(os.path.join(root, e)))
            self.name_project.addItems(entries)
        self.name_project.blockSignals(False)

    def _on_project_changed(self, _):
        self._update_tree_from_combo()

    def _update_tree_from_combo(self):
        goods = self._get_goods_path()
        if goods:
            self._set_tree_root(str(goods))

    def _get_project_root(self) -> Path | None:
        name = self.name_project.currentText().strip()
        return Path(self._get_projects_root()) / name if name else None

    def _get_goods_path(self) -> Path | None:
        pr = self._get_project_root()
        if pr is None:
            return None
        g = pr / "goods"
        g.mkdir(parents=True, exist_ok=True)
        return g

    def _get_memory_bank_dir(self) -> Path | None:
        pr = self._get_project_root()
        if pr is None:
            return None
        d = pr / "memory_bank"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_template_point_path(self) -> Path | None:
        pr = self._get_project_root()
        return pr / TEMPLATE_POINT_FILENAME if pr else None

    # ======================================================================= #
    #  TREE VIEW                                                               #
    # ======================================================================= #
    def _set_tree_root(self, path: str):
        idx = self._fs_model.setRootPath(path)
        self.treeView_img.setRootIndex(idx)
        self.treeView_img.expandAll()

    def _set_data_buttons_enabled(self, enabled: bool):
        self.btn_start_get_data.setEnabled(enabled)
        self.btn_stop_get_data.setEnabled(enabled)
        self.build_memory_bank.setEnabled(enabled)

    def _on_tree_selection_changed(self, selected, _):
        indexes = selected.indexes()
        if not indexes:
            return
        file_path = self._fs_model.filePath(indexes[0])
        if not os.path.isfile(file_path):
            return
        if os.path.splitext(file_path)[1].lower() not in \
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
            return
        self._current_image_path = file_path
        self._select_mode        = None
        self._template_point_1   = None
        self._template_point_2   = None
        self._update_select_buttons_state()
        self._display_image(file_path)

    # ======================================================================= #
    #  DISPLAY HELPERS                                                         #
    # ======================================================================= #
    def _get_scale_offset(self, orig_w, orig_h):
        lw, lh   = self.img_show.width(), self.img_show.height()
        scale    = min(lw / orig_w, lh / orig_h)
        sw, sh   = int(orig_w * scale), int(orig_h * scale)
        ox, oy   = (lw - sw) // 2, (lh - sh) // 2
        return scale, sw, sh, ox, oy

    def _display_image(self, file_path: str,
                       point1: QPoint | None = None,
                       point2: QPoint | None = None):
        pm = QPixmap(file_path)
        if pm.isNull():
            return
        ow, oh = pm.width(), pm.height()
        scale, sw, sh, ox, oy = self._get_scale_offset(ow, oh)
        spm = pm.scaled(sw, sh,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        if point1 is not None:
            px1, py1 = int(point1.x() - ox), int(point1.y() - oy)
            p = QPainter(spm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            if point2 is None:
                p.setPen(QPen(QColor(0, 255, 0), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPoint(px1, py1), 6, 6)
                p.drawLine(px1 - 10, py1, px1 + 10, py1)
                p.drawLine(px1, py1 - 10, px1, py1 + 10)
            else:
                px2, py2 = int(point2.x() - ox), int(point2.y() - oy)
                rect = QRect(QPoint(px1, py1), QPoint(px2, py2)).normalized()
                p.setBrush(QBrush(QColor(0, 255, 0, 40)))
                p.setPen(QPen(QColor(0, 255, 0), 2))
                p.drawRect(rect)
                p.setBrush(QBrush(QColor(0, 255, 0)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPoint(px1, py1), 4, 4)
                p.drawEllipse(QPoint(px2, py2), 4, 4)
            p.end()
        self.img_show.setPixmap(spm)

    def _display_cv2_image(self, img_bgr: np.ndarray):
        rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pm    = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.img_show.width(), self.img_show.height(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self.img_show.setPixmap(scaled)

    # ======================================================================= #
    #  COORDINATE CONVERSION                                                   #
    # ======================================================================= #
    def _label_to_img(self, lp: QPoint) -> tuple[int, int]:
        if not self._current_image_path:
            return lp.x(), lp.y()
        pm = QPixmap(self._current_image_path)
        if pm.isNull():
            return lp.x(), lp.y()
        scale, _, _, ox, oy = self._get_scale_offset(pm.width(), pm.height())
        rx = max(0.0, min((lp.x() - ox) / scale, float(pm.width()  - 1)))
        ry = max(0.0, min((lp.y() - oy) / scale, float(pm.height() - 1)))
        return int(round(rx)), int(round(ry))

    # ======================================================================= #
    #  TEMPLATE SELECT                                                         #
    # ======================================================================= #
    def _update_select_buttons_state(self):
        sel = self._select_mode is not None
        self.btn_select_template.setEnabled(not sel)
        self.btn_cancle_select.setEnabled(sel)

    def _on_select_template(self):
        if self._current_image_path is None:
            QtWidgets.QMessageBox.warning(self, "Chưa chọn ảnh",
                                          "Vui lòng chọn ảnh từ danh sách trước.")
            return
        self._select_mode      = "waiting_first"
        self._template_point_1 = None
        self._template_point_2 = None
        self._update_select_buttons_state()
        self._display_image(self._current_image_path)

    def _on_cancel_select(self):
        self._select_mode      = None
        self._template_point_1 = None
        self._template_point_2 = None
        self._update_select_buttons_state()
        if self._current_image_path:
            self._display_image(self._current_image_path)

    def _on_image_clicked(self, point: QPoint):
        if self._select_mode is None:
            return
        if self._select_mode == "waiting_first":
            self._template_point_1 = point
            self._select_mode      = "waiting_second"
            self._update_select_buttons_state()
            self._display_image(self._current_image_path, point1=point)
        elif self._select_mode == "waiting_second":
            self._template_point_2 = point
            self._display_image(self._current_image_path,
                                point1=self._template_point_1, point2=point)
            self._save_template_point(self._template_point_1, point)
            self._select_mode      = None
            self._template_point_1 = None
            self._template_point_2 = None
            self._update_select_buttons_state()

    def _save_template_point(self, p1: QPoint, p2: QPoint):
        json_path = self._get_template_point_path()
        if json_path is None:
            return
        x1, y1 = self._label_to_img(p1)
        x2, y2 = self._label_to_img(p2)
        rx, ry  = min(x1, x2), min(y1, y2)
        rw, rh  = abs(x2 - x1), abs(y2 - y1)
        data = {"image_path": self._current_image_path,
                "point1": {"x": x1, "y": y1}, "point2": {"x": x2, "y": y2},
                "rect":   {"x": rx, "y": ry,  "w": rw,  "h": rh}}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        QtWidgets.QMessageBox.information(
            self, "Lưu thành công",
            f"Rect: x={rx}, y={ry}, w={rw}, h={rh}"
        )

    # ======================================================================= #
    #  LIQUID LEVEL SETUP                                                      #
    # ======================================================================= #
    def _on_setup_liquid(self):
        if self._current_image_path is None:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh.")
            return

        json_path = self._get_template_point_path()
        if json_path is None or not json_path.exists():
            QtWidgets.QMessageBox.warning(self, "Chưa chọn vùng ROI",
                                          "Dùng 'Select Template' để khoanh vùng mặt nước trước.")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            tp = json.load(f)
        rect = tp.get("rect", {})
        if not rect or rect.get("w", 0) <= 0 or rect.get("h", 0) <= 0:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Vùng chọn không hợp lệ.")
            return

        x, y, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
        img = cv2.imread(self._current_image_path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Lỗi",
                                          f"Không đọc được ảnh:\n{self._current_image_path}")
            return

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = gray[y: y + rh, x: x + rw]

        pr = self._get_project_root()
        if pr is None:
            return

        self.btn_setup_liquid.setEnabled(False)

        self._liquid_worker = LiquidSetupWorker(
            img_bgr       = img,
            template_gray = template,
            x=x, w=rw, h=rh,
            project_root  = pr,
        )
        self._liquid_worker.detect_done.connect(self._on_liquid_detect_done)
        self._liquid_worker.start()

    def _on_liquid_detect_done(self, success: bool, fill_ratio: float,
                               display_img, error_msg: str):
        self.btn_setup_liquid.setEnabled(True)

        if not success:
            QtWidgets.QMessageBox.warning(self, "Detect thất bại", error_msg)
            self._liquid_worker = None
            return

        if display_img is not None:
            self._display_cv2_image(display_img)

        tolerance, ok = QtWidgets.QInputDialog.getDouble(
            self, "Dung sai mực nước",
            f"Fill ratio đo được: {fill_ratio:.1f}%\n\nNhập dung sai ± (%)",
            value=10.0, min=0.5, max=50.0, decimals=1
        )
        if not ok:
            self._liquid_worker = None
            return

        final_img = self._liquid_worker.save_final(tolerance)
        if final_img is not None:
            self._display_cv2_image(final_img)

        min_f = max(0.0,   fill_ratio - tolerance)
        max_f = min(100.0, fill_ratio + tolerance)
        QtWidgets.QMessageBox.information(
            self, "Setup hoàn tất",
            f"Fill ratio  : {fill_ratio:.1f}%\n"
            f"Dung sai    : ± {tolerance:.1f}%\n"
            f"Ngưỡng OK   : [{min_f:.1f}% – {max_f:.1f}%]\n\n"
            f"(U2Net mask đã được dùng để detect mực nước)"
        )
        self._liquid_worker = None

    # ======================================================================= #
    #  CAMERA                                                                  #
    # ======================================================================= #
    def init_camera(self):
        if self.is_initialized:
            return
        pr = self._get_project_root()
        if pr is None:
            print("[GetData] ⚠️  Chưa chọn project.")
            return
        try:
            self.data_manager = GetDataManager(project_root=pr)
            self.data_manager.start()
            self.is_initialized = True
            print(f"[GetData] ✅ Camera khởi tạo — project: {pr.name}")
        except Exception as e:
            print(f"[GetData] ❌ Khởi tạo thất bại: {e}")

    def start_get_data(self):
        if self.is_saving:
            return
        save_dir = self._get_goods_path()
        if not save_dir:
            return
        self.init_camera()
        if not self.is_initialized:
            return
        self._set_tree_root(str(save_dir))
        self.data_manager.set_save_dir(str(save_dir))
        self.data_manager.is_saving = True
        self.is_saving = True
        print(f"[GetData] ▶ Thu thập → {save_dir}")

    def stop_get_data(self):
        if not self.is_saving:
            return
        self.data_manager.is_saving = False
        self.is_saving = False
        print("[GetData] ⏹ Dừng thu thập.")

    # ======================================================================= #
    #  BUILD MEMORY BANK                                                       #
    # ======================================================================= #
    def build_memory(self):
        if self._build_worker and self._build_worker.isRunning():
            return
        goods_dir  = self._get_goods_path()
        output_dir = self._get_memory_bank_dir()
        if goods_dir is None or output_dir is None:
            return
        self.stop_get_data()
        self._set_data_buttons_enabled(False)
        self._build_worker = BuildBankWorker(goods_dir, output_dir)
        self._build_worker.finished.connect(self._on_build_finished)
        self._build_worker.start()

    def _on_build_finished(self, success: bool, message: str):
        print(f"[GetData] {message}")
        self._set_data_buttons_enabled(True)

    # ======================================================================= #
    #  MISC                                                                    #
    # ======================================================================= #
    def refresh_projects(self):
        current = self.name_project.currentText()
        self._populate_project_combo()
        idx = self.name_project.findText(current)
        if idx >= 0:
            self.name_project.setCurrentIndex(idx)
        self._update_tree_from_combo()

    def stop_all_threads(self):
        self.stop_get_data()
        if self._build_worker and self._build_worker.isRunning():
            self._build_worker.quit()
            self._build_worker.wait(3000)
        if self._liquid_worker and self._liquid_worker.isRunning():
            self._liquid_worker.quit()
            self._liquid_worker.wait(3000)
        if self.data_manager:
            self.data_manager.stop()
            self.data_manager.join()
            self.data_manager   = None
            self.is_initialized = False
        print("[GetData] ✅ Tất cả thread đã dừng.")

    def closeEvent(self, event):
        self.stop_all_threads()
        super().closeEvent(event)


# =========================================================================== #
#  GET DATA MANAGER                                                            #
# =========================================================================== #
class GetDataManager(threading.Thread):
    def __init__(self, project_root: Path):
        super().__init__(daemon=True, name="GetDataManager")
        self.project_root   = project_root
        self.frames_queue   = queue.Queue(maxsize=1)
        self._stop_event    = threading.Event()
        self.is_saving      = False
        self.save_dir       = None
        self._trigger_count = 0

    def set_save_dir(self, path: str):
        self.save_dir       = path
        self._trigger_count = self._scan_last_index(path)
        if self._trigger_count > 0:
            print(f"[GetDataManager] 🔄 Resume từ trigger_{self._trigger_count:04d}")
        else:
            print("[GetDataManager] 🆕 Bắt đầu từ 0001")

    def _load_project_info(self) -> dict | None:
        json_path = self.project_root / PROJECT_INFO_FILENAME
        if not json_path.exists():
            print(f"[GetDataManager] ⚠️  Không tìm thấy {json_path}")
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _scan_last_index(directory: str) -> int:
        if not os.path.isdir(directory):
            return 0
        pattern   = re.compile(r"^cam\d+_(\d+)\.jpg$", re.IGNORECASE)
        max_index = 0
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                max_index = max(max_index, int(m.group(1)))
        return max_index

    def stop(self):
        self._stop_event.set()

    def run(self):
        thread_camera = None
        try:
            infor_project = self._load_project_info()
            thread_camera = BatchCamera(self.frames_queue, self._stop_event, infor_project)
            thread_camera.start()
            print("[GetDataManager] Camera started.")
            while not self._stop_event.is_set():
                try:
                    _, frames = self.frames_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if self.is_saving and self.save_dir:
                    self._save_trigger(frames)
        except Exception as e:
            print(f"[GetDataManager] ❌ {e}")
        finally:
            if thread_camera:
                thread_camera.join()
            print("[GetDataManager] Stopped.")

    def _save_trigger(self, frames: list):
        self._trigger_count += 1
        for cam_id, frame in enumerate(frames):
            if frame is None:
                continue
            fname = f"cam{cam_id+1}_{self._trigger_count:04d}.jpg"
            path  = os.path.join(self.save_dir, fname)
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[GetDataManager] 💾 trigger_{self._trigger_count:04d}")


    