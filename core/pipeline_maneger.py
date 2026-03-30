<<<<<<< HEAD
# import threading
# import queue
# import time
# from hardware.camera.batch_camera import BatchCamera

# class PipelineManager(threading.Thread):

#     def __init__(self, show_queue):

#         super().__init__()
#         self.frames_queue = queue.Queue(maxsize=1)
#         self.show_queue = show_queue
    
        
#         self.running = True


  
#     def run(self):
#         thread_camera = None
#         stop_event = threading.Event()
#         try:
#             thread_camera = BatchCamera(self.frames_queue, stop_event)
#             thread_camera.start()
            
            
            

#             print("[Pipeline] Started")

#             while self.running:
#                 try:
#                     trigger_id,frames = self.frames_queue.get(timeout=1)



#                     for cam_id, frame in enumerate(frames):
                        


                    
#                         try:
#                             self.show_queue[str(cam_id+1)].put_nowait(frame)
#                         except queue.Full:
#                             try:
#                                 self.show_queue[str(cam_id+1)].get_nowait()
#                                 self.show_queue[str(cam_id+1)].put_nowait(frame)
#                             except queue.Empty:
#                                 pass


                
#                 except queue.Empty:
#                     pass                                            
#         finally:
#             if thread_camera is not None:
                
#                 stop_event.set()
#                 thread_camera = None

#                 print("[Pipeline] Stopping camera thread...")



        




#     def stop(self):
#         self.running = False

#         print("[Pipeline] Stopping...")

        



#         print("[Pipeline] Stopped")


# import threading
# import queue
# import time
# from hardware.camera.batch_camera import BatchCamera
# from core.body_inspection import BodyWorker


# class PipelineManager(threading.Thread):
#     """
#     Pipeline:
#         BatchCamera
#             │ frames_queue  →  (trigger_id, [img1,img2,img3,img4,img5])
#             ▼
#         PipelineManager.run()
#             ├── show_queue[cam_key]  ← raw frame mỗi cam (hiển thị ngay)
#             └── body_in_queue       ← (trigger_id, [img1,img2,img3,img4])  4 ảnh đầu
#                     │
#                     ▼
#                 BodyWorker  (1 forward pass cho cả batch 4 ảnh)
#                     │ result_queue  →  (trigger_id, list[dict])
#                     ▼
#                 _result_loop()
#                     └── show_queue[cam_key] ← kết quả OK/NG (tuỳ chọn overlay)
#     """

#     def __init__(self, show_queue, threshold: float = 14.0):
#         super().__init__(daemon=True, name="PipelineManager")
#         self.show_queue  = show_queue    # dict[str, queue.Queue]
#         self.threshold   = threshold
#         self.running     = True

#         # Queues nội bộ
#         self.frames_queue  = queue.Queue(maxsize=1)   # BatchCamera  → Pipeline
#         self.body_in_queue = queue.Queue(maxsize=4)   # Pipeline     → BodyWorker
#         self.result_queue  = queue.Queue()            # BodyWorker   → Pipeline

#         self._stop_event = threading.Event()

#     # ----------------------------------------------------------------------- #
#     def run(self):
#         thread_camera = None
#         thread_body   = None

#         try:
#             # ── Khởi động BatchCamera ───────────────────────────────────────
#             thread_camera = BatchCamera(self.frames_queue, self._stop_event)
#             thread_camera.start()

#             # ── Khởi động BodyWorker ────────────────────────────────────────
#             thread_body = BodyWorker(
#                 frame_input_queue   = self.body_in_queue,
#                 result_output_queue = self.result_queue,
#                 stop_event          = self._stop_event,
#                 threshold           = self.threshold,
#             )
#             thread_body.start()

#             # # ── Thread đọc kết quả từ BodyWorker ───────────────────────────
#             # result_reader = threading.Thread(
#             #     target=self._result_loop,
#             #     daemon=True,
#             #     name="ResultReader",
#             # )
#             # result_reader.start()

#             print("[Pipeline] Started")

#             # ── Vòng lặp chính ─────────────────────────────────────────────
#             while self.running:
#                 try:
#                     trigger_id, frames = self.frames_queue.get(timeout=1)
#                     # frames = [img1, img2, img3, img4, img5]
#                 except queue.Empty:
#                     continue

#                 # # 1. Đưa raw frame lên show_queue để hiển thị ngay
#                 # for cam_id, frame in enumerate(frames):
#                 #     cam_key = str(cam_id + 1)
#                 #     self._put_drop(self.show_queue.get(cam_key), frame)

#                 # 2. Gửi 4 ảnh đầu vào BodyWorker
#                 #    Giữ nguyên vị trí (cam_id) kể cả khi ảnh là None
#                 #    BodyWorker sẽ bỏ qua None, kết quả trả về chỉ có cam có ảnh thật
#                 batch_4 = frames[:4]   # list 4 phần tử, mỗi phần tử là PIL.Image | None
#                 self._put_drop(self.body_in_queue, (trigger_id, batch_4))

#         except Exception as e:
#             print(f"[Pipeline] ❌ Lỗi: {e}")

#         finally:
#             self._cleanup(thread_camera, thread_body)

#     # ----------------------------------------------------------------------- #
#     def _result_loop(self):
#         """Đọc kết quả từ BodyWorker → log + gửi show_queue nếu cần."""
#         print("[ResultReader] ▶ Bắt đầu …")

#         while not self._stop_event.is_set():
#             try:
#                 item = self.result_queue.get(timeout=0.5)
#             except queue.Empty:
#                 continue

#             if item is None:
#                 break

#             trigger_id, results = item

#             if isinstance(results, dict) and "error" in results:
#                 print(f"[ResultReader] ⚠️  trigger={trigger_id} – {results['error']}")
#             else:
#                 # results[cam_id] = dict | None
#                 for cam_id, r in enumerate(results):
#                     if r is None:
#                         continue              # cam này không có ảnh
#                     label   = "✅ OK" if r["is_ok"] else "❌ NG"
#                     cam_key = str(cam_id + 1)
#                     print(
#                         f"[ResultReader] trigger={trigger_id} cam={cam_key} | "
#                         f"{label} | score={r['score']:.4f} | {r['time_ms']:.1f}ms"
#                     )
#                     # Tuỳ chọn: gửi kết quả lên show_queue để UI overlay
#                     # self._put_drop(self.show_queue.get(cam_key), r)

#             self.result_queue.task_done()

#         print("[ResultReader] 🛑 Kết thúc.")

#     # ----------------------------------------------------------------------- #
#     @staticmethod
#     def _put_drop(q, item):
#         """Đặt item vào queue; nếu full thì drop cũ, đặt mới."""
#         if q is None:
#             return
#         try:
#             q.put_nowait(item)
#         except queue.Full:
#             try:
#                 q.get_nowait()
#             except queue.Empty:
#                 pass
#             try:
#                 q.put_nowait(item)
#             except queue.Full:
#                 pass

#     # ----------------------------------------------------------------------- #
#     def _cleanup(self, thread_camera, thread_body):
#         print("[Pipeline] Dọn dẹp …")
#         self._stop_event.set()

#         try:
#             self.body_in_queue.put_nowait(None)   # sentinel cho BodyWorker
#         except queue.Full:
#             pass

#         if thread_camera:
#             thread_camera.join(timeout=3)
#             print("[Pipeline] Camera thread đã dừng.")

#         if thread_body:
#             thread_body.join(timeout=5)
#             print("[Pipeline] BodyWorker đã dừng.")

#         print("[Pipeline] Stopped")

#     # ----------------------------------------------------------------------- #
#     def stop(self):
#         print("[Pipeline] Stopping …")
#         self.running = False



=======
>>>>>>> 5fe2763 (update 2703)
import threading
import queue
import time
import cv2
import numpy as np
from hardware.camera.batch_camera import BatchCamera
<<<<<<< HEAD
from core.body_inspection import BodyWorker
=======
from hardware.sorting.sorting_actuator import SortingActuator
from core.body_inspection import BodyWorker, draw_anomaly_overlay
>>>>>>> 5fe2763 (update 2703)


class PipelineManager(threading.Thread):
    """
    Pipeline:
        BatchCamera
            │ frames_queue → (trigger_id, [img1,img2,img3,img4,img5])
            ▼
        PipelineManager.run()
<<<<<<< HEAD
            ├── show_queue[cam_key] ← raw frame (hiển thị ngay, không chờ AI)
=======
            ├── show_queue[cam_key] ← (frame_bgr, None)   raw frame (hiển thị ngay)
>>>>>>> 5fe2763 (update 2703)
            └── body_in_queue      ← (trigger_id, [img1,img2,img3,img4])
                    │
                    ▼
                BodyWorker
                    │ result_queue → (trigger_id, list[dict|None])
                    ▼
                _result_loop()
<<<<<<< HEAD
                    └── show_queue[cam_key] ← frame đã overlay OK/NG
    """

    def __init__(self, show_queue, threshold: float = 14.0):
        super().__init__(daemon=True, name="PipelineManager")
        self.show_queue  = show_queue   # dict[str, queue.Queue]  – key "1".."5"
=======
                    ├── show_queue[cam_key] ← (frame_overlay, is_ok)
                    └── sorting_queue       ← batch_ok (bool)
                                │
                                ▼
                        SortingActuator
                            └── chờ trigger vật lý → lấy batch_ok → log/actuate

    show_queue nhận tuple (frame_bgr: np.ndarray, is_ok: bool | None)
        is_ok = None  → raw frame chưa có kết quả AI
        is_ok = True  → AI phán OK
        is_ok = False → AI phán NG
    """

    def __init__(self, show_queue, threshold: float = 18):
        super().__init__(daemon=True, name="PipelineManager")
        self.show_queue  = show_queue   # dict[str, queue.Queue] – key "1".."5"
>>>>>>> 5fe2763 (update 2703)
        self.threshold   = threshold
        self.running     = True

        self.frames_queue  = queue.Queue(maxsize=1)
        self.body_in_queue = queue.Queue(maxsize=4)
        self.result_queue  = queue.Queue()
<<<<<<< HEAD
        self._stop_event   = threading.Event()

        # Giữ raw frame mới nhất mỗi cam để overlay kết quả AI lên sau
        self._last_frames  = {}          # cam_key → numpy.ndarray

    # ----------------------------------------------------------------------- #
    def run(self):
        thread_camera = None
        thread_body   = None
=======
        self.sorting_queue = queue.Queue(maxsize=20)
        self._stop_event   = threading.Event()

        # Giữ raw frame BGR mới nhất mỗi cam để overlay kết quả AI lên sau
        self._last_frames = {}   # cam_key → numpy.ndarray (BGR)

    # ----------------------------------------------------------------------- #
    def run(self):
        thread_camera  = None
        thread_body    = None
        thread_sorting = None
>>>>>>> 5fe2763 (update 2703)

        try:
            # ── BatchCamera ─────────────────────────────────────────────────
            try:
                thread_camera = BatchCamera(self.frames_queue, self._stop_event)
                thread_camera.start()
                print("[Pipeline] BatchCamera started.")
            except Exception as e:
                print(f"[Pipeline] ❌ BatchCamera khởi tạo thất bại: {e}")
                self.running = False
                return

<<<<<<< HEAD
=======
            # ── SortingActuator ─────────────────────────────────────────────
            try:
                thread_sorting = SortingActuator(self.sorting_queue, self._stop_event)
                thread_sorting.start()
                print("[Pipeline] SortingActuator started.")
            except Exception as e:
                print(f"[Pipeline] ❌ SortingActuator khởi tạo thất bại: {e}")
                self.running = False
                return

>>>>>>> 5fe2763 (update 2703)
            # ── BodyWorker ──────────────────────────────────────────────────
            try:
                thread_body = BodyWorker(
                    frame_input_queue   = self.body_in_queue,
                    result_output_queue = self.result_queue,
                    stop_event          = self._stop_event,
                    threshold           = self.threshold,
                )
                thread_body.start()
                print("[Pipeline] BodyWorker started.")
            except Exception as e:
                print(f"[Pipeline] ❌ BodyWorker khởi tạo thất bại: {e}")
                self.running = False
                return

            # ── ResultReader thread ─────────────────────────────────────────
            result_reader = threading.Thread(
                target=self._result_loop,
                daemon=True,
                name="ResultReader",
            )
            result_reader.start()

            print("[Pipeline] Started")

            # ── Vòng lặp chính ─────────────────────────────────────────────
            while self.running:
                try:
                    trigger_id, frames = self.frames_queue.get(timeout=1)
                except queue.Empty:
                    continue

<<<<<<< HEAD
                # 1. Show raw frame ngay lên UI (không chờ AI)
=======
                # 1. Show raw frame ngay lên UI – is_ok=None (chưa có kết quả AI)
>>>>>>> 5fe2763 (update 2703)
                for cam_id, frame in enumerate(frames):
                    cam_key = str(cam_id + 1)
                    if frame is None:
                        continue
<<<<<<< HEAD
                    self._last_frames[cam_key] = frame          # lưu lại để overlay
                    self._put_drop(self.show_queue.get(cam_key), frame)

                # 2. Gửi 4 ảnh đầu vào BodyWorker
=======
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._last_frames[cam_key] = frame_bgr
                    self._put_drop(self.show_queue.get(cam_key), (frame_bgr, None))

                # 2. Gửi 4 ảnh đầu vào BodyWorker – RGB gốc, KHÔNG dùng frame_bgr
>>>>>>> 5fe2763 (update 2703)
                batch_4 = frames[:4]
                self._put_drop(self.body_in_queue, (trigger_id, batch_4))

        except Exception as e:
            print(f"[Pipeline] ❌ Lỗi: {e}")

        finally:
<<<<<<< HEAD
            self._cleanup(thread_camera, thread_body)

    # ----------------------------------------------------------------------- #
    def _result_loop(self):
        """Nhận kết quả AI → overlay lên frame → gửi show_queue."""
=======
            self._cleanup(thread_camera, thread_body, thread_sorting)

    # ----------------------------------------------------------------------- #
    def _result_loop(self):
        """Nhận kết quả AI → overlay → gửi show_queue + sorting_queue."""
>>>>>>> 5fe2763 (update 2703)
        print("[ResultReader] ▶ Bắt đầu …")

        while not self._stop_event.is_set():
            try:
                item = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            trigger_id, results = item

            if isinstance(results, dict) and "error" in results:
                print(f"[ResultReader] ⚠️  trigger={trigger_id} – {results['error']}")
                self.result_queue.task_done()
                continue

<<<<<<< HEAD
            # results[cam_id] = dict | None  – thứ tự theo cam_id
=======
            # Tính is_ok cho toàn batch: tất cả cam đều OK mới là OK
            batch_ok = all(r["is_ok"] for r in results if r is not None)

            # Gửi kết quả batch vào sorting_queue để SortingActuator xử lý
            try:
                self.sorting_queue.put_nowait(batch_ok)
            except queue.Full:
                print("[ResultReader] ⚠️  sorting_queue full – drop kết quả cũ nhất")
                try:
                    self.sorting_queue.get_nowait()
                except queue.Empty:
                    pass
                self.sorting_queue.put_nowait(batch_ok)

            # Overlay từng cam + gửi show_queue
>>>>>>> 5fe2763 (update 2703)
            for cam_id, r in enumerate(results):
                if r is None:
                    continue

                cam_key = str(cam_id + 1)
<<<<<<< HEAD
                label   = "✅ OK" if r["is_ok"] else "❌ NG"
                print(
                    f"[ResultReader] trigger={trigger_id} cam={cam_key} | "
                    f"{label} | score={r['score']:.4f} | {r['time_ms']:.1f}ms"
                )

                # Overlay kết quả lên raw frame mới nhất
                frame = self._last_frames.get(cam_key)
                if frame is not None:
                    overlay = self._draw_result(frame, r)
                    self._put_drop(self.show_queue.get(cam_key), overlay)
=======
                is_ok   = r["is_ok"]
                label   = "✅ OK" if is_ok else "❌ NG"
                boxes   = r["anomaly_info"]["boxes"]
                print(
                    f"[ResultReader] trigger={trigger_id} cam={cam_key} | "
                    f"{label} | score={r['score']:.4f} | "
                    f"boxes={len(boxes)} | {r['time_ms']:.1f}ms"
                )

                frame_bgr = self._last_frames.get(cam_key)
                if frame_bgr is not None:
                    overlay = self._draw_result(frame_bgr, r)
                    self._put_drop(self.show_queue.get(cam_key), (overlay, is_ok))

            print(
                f"[ResultReader] trigger={trigger_id} "
                f"BATCH → {'✅ OK' if batch_ok else '❌ NG'}"
            )
>>>>>>> 5fe2763 (update 2703)

            self.result_queue.task_done()

        print("[ResultReader] 🛑 Kết thúc.")

    # ----------------------------------------------------------------------- #
    @staticmethod
<<<<<<< HEAD
    def _draw_result(frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Vẽ kết quả OK/NG lên frame.
        Trả về frame mới (không sửa in-place).
        """
        img = frame.copy()
        is_ok   = result["is_ok"]
        score   = result["score"]
        label   = "OK" if is_ok else "NG"
        color   = (0, 255, 0) if is_ok else (0, 0, 255)   # BGR: xanh / đỏ

        h, w    = img.shape[:2]
        margin  = 20

        # Nền mờ góc trên-trái
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w // 3, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        # Text kết quả
        cv2.putText(img, label,
                    (margin, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)

        # Text score nhỏ hơn
        cv2.putText(img, f"score: {score:.3f}",
=======
    def _draw_result(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        """
        Vẽ heatmap anomaly + bounding boxes + label OK/NG lên frame BGR.
        Trả về frame mới (không sửa in-place).
        """
        # 1. Overlay heatmap + bounding boxes (chỉ khi NG)
        if not result["is_ok"] and result.get("anomaly_info"):
            img = draw_anomaly_overlay(
                frame_bgr    = frame_bgr,
                anomaly_info = result["anomaly_info"],
                alpha        = 0.4,
            )
        else:
            img = frame_bgr.copy()

        # 2. Label OK / NG + score góc trên-trái
        is_ok  = result["is_ok"]
        label  = "OK" if is_ok else "NG"
        color  = (0, 255, 0) if is_ok else (0, 0, 255)
        h, w   = img.shape[:2]
        margin = 20

        bg = img.copy()
        cv2.rectangle(bg, (0, 0), (w // 3, 90), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.45, img, 0.55, 0, img)

        cv2.putText(img, label,
                    (margin, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
        cv2.putText(img, f"score: {result['score']:.3f}",
>>>>>>> 5fe2763 (update 2703)
                    (margin, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        return img

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _put_drop(q, item):
<<<<<<< HEAD
        """Đặt item vào queue; nếu full thì drop cũ rồi đặt mới."""
=======
        """
        Đặt item vào queue; nếu full thì drop cũ rồi đặt mới.
        item = (frame_bgr, is_ok)  – is_ok: True | False | None
        """
>>>>>>> 5fe2763 (update 2703)
        if q is None:
            return
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    # ----------------------------------------------------------------------- #
<<<<<<< HEAD
    def _cleanup(self, thread_camera, thread_body):
=======
    def _cleanup(self, thread_camera, thread_body, thread_sorting=None):
>>>>>>> 5fe2763 (update 2703)
        print("[Pipeline] Dọn dẹp …")
        self._stop_event.set()

        try:
            self.body_in_queue.put_nowait(None)
        except queue.Full:
            pass

        if thread_camera:
            thread_camera.join(timeout=3)
            print("[Pipeline] Camera thread đã dừng.")

        if thread_body:
            thread_body.join(timeout=5)
            print("[Pipeline] BodyWorker đã dừng.")

<<<<<<< HEAD
=======
        if thread_sorting:
            thread_sorting.join(timeout=3)
            print("[Pipeline] SortingActuator đã dừng.")

>>>>>>> 5fe2763 (update 2703)
        print("[Pipeline] Stopped")

    # ----------------------------------------------------------------------- #
    def stop(self):
        print("[Pipeline] Stopping …")
        self.running = False