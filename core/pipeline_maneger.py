import threading
import queue
import time
import cv2
import numpy as np
from hardware.camera.batch_camera import BatchCamera
from hardware.sorting.sorting_actuator import SortingActuator
from core.body_inspection import BodyWorker, draw_anomaly_overlay


class PipelineManager(threading.Thread):
    """
    Pipeline:
        BatchCamera
            │ frames_queue → (trigger_id, [img1,img2,img3,img4,img5])
            ▼
        PipelineManager.run()
            ├── show_queue[cam_key] ← (frame_bgr, None)   raw frame (hiển thị ngay)
            └── body_in_queue      ← (trigger_id, [img1,img2,img3,img4])
                    │
                    ▼
                BodyWorker
                    │ result_queue → (trigger_id, list[dict|None])
                    ▼
                _result_loop()
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

    def __init__(self, show_queue, infor_project, threshold: float = 18):
        super().__init__(daemon=True, name="PipelineManager")
        self.show_queue  = show_queue   # dict[str, queue.Queue] – key "1".."5"
        self.infor_project = infor_project
        self.threshold   = self.infor_project.get("settings", {}).get("threshold", threshold)
        self.running     = True

        self.frames_queue  = queue.Queue(maxsize=1)
        self.body_in_queue = queue.Queue(maxsize=4)
        self.result_queue  = queue.Queue()
        self.sorting_queue = queue.Queue(maxsize=20)
        self._stop_event   = threading.Event()

        # Giữ raw frame BGR mới nhất mỗi cam để overlay kết quả AI lên sau
        self._last_frames = {}   # cam_key → numpy.ndarray (BGR)

    # ----------------------------------------------------------------------- #
    def run(self):
        thread_camera  = None
        thread_body    = None
        thread_sorting = None

        try:
            # ── BatchCamera ─────────────────────────────────────────────────
            try:
                thread_camera = BatchCamera(self.frames_queue, self._stop_event, self.infor_project)
                thread_camera.start()
                print("[Pipeline] BatchCamera started.")
            except Exception as e:
                print(f"[Pipeline] ❌ BatchCamera khởi tạo thất bại: {e}")
                self.running = False
                return

            # ── SortingActuator ─────────────────────────────────────────────
            try:
                thread_sorting = SortingActuator(self.sorting_queue, self._stop_event)
                thread_sorting.start()
                print("[Pipeline] SortingActuator started.")
            except Exception as e:
                print(f"[Pipeline] ❌ SortingActuator khởi tạo thất bại: {e}")
                self.running = False
                return

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

                # 1. Show raw frame ngay lên UI – is_ok=None (chưa có kết quả AI)
                for cam_id, frame in enumerate(frames):
                    cam_key = str(cam_id + 1)
                    if frame is None:
                        continue
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._last_frames[cam_key] = frame_bgr
                    self._put_drop(self.show_queue.get(cam_key), (frame_bgr, None))

                # 2. Gửi 4 ảnh đầu vào BodyWorker – RGB gốc, KHÔNG dùng frame_bgr
                batch_4 = frames[:4]
                self._put_drop(self.body_in_queue, (trigger_id, batch_4))

        except Exception as e:
            print(f"[Pipeline] ❌ Lỗi: {e}")

        finally:
            self._cleanup(thread_camera, thread_body, thread_sorting)

    # ----------------------------------------------------------------------- #
    def _result_loop(self):
        """Nhận kết quả AI → overlay → gửi show_queue + sorting_queue."""
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
            for cam_id, r in enumerate(results):
                if r is None:
                    continue

                cam_key = str(cam_id + 1)
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

            self.result_queue.task_done()

        print("[ResultReader] 🛑 Kết thúc.")

    # ----------------------------------------------------------------------- #
    # @staticmethod
    # def _draw_result(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    #     """
    #     Vẽ heatmap anomaly + bounding boxes + label OK/NG lên frame BGR.
    #     Trả về frame mới (không sửa in-place).
    #     """
    #     # 1. Overlay heatmap + bounding boxes (chỉ khi NG)
    #     if not result["is_ok"] and result.get("anomaly_info"):
    #         img = draw_anomaly_overlay(
    #             frame_bgr    = frame_bgr,
    #             anomaly_info = result["anomaly_info"],
    #             alpha        = 0.4,
    #         )
    #     else:
    #         img = frame_bgr.copy()

    #     # 2. Label OK / NG + score góc trên-trái
    #     is_ok  = result["is_ok"]
    #     label  = "OK" if is_ok else "NG"
    #     color  = (0, 255, 0) if is_ok else (0, 0, 255)
    #     h, w   = img.shape[:2]
    #     margin = 20

    #     bg = img.copy()
    #     cv2.rectangle(bg, (0, 0), (w // 3, 90), (0, 0, 0), -1)
    #     cv2.addWeighted(bg, 0.45, img, 0.55, 0, img)

    #     cv2.putText(img, label,
    #                 (margin, 60),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
    #     cv2.putText(img, f"score: {result['score']:.3f}",
    #                 (margin, 85),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    #     return img


    @staticmethod
    def _draw_result(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        """
        Vẽ theo thứ tự:
          1. Mask vùng vật thể (U2Net) — viền xanh lá + tô bán trong suốt
          2. Heatmap anomaly + bounding boxes (chỉ khi NG)
          3. Label OK/NG + score góc trên-trái
        """
        from core.body_inspection import draw_anomaly_overlay, draw_object_mask

        # 1. Vẽ vùng vật thể từ U2Net mask (nếu có)
        object_mask = result.get("object_mask")
        if object_mask is not None:
            img = draw_object_mask(frame_bgr, object_mask)
        else:
            img = frame_bgr.copy()

        # 2. Overlay heatmap + bounding boxes (chỉ khi NG)
        if not result["is_ok"] and result.get("anomaly_info"):
            img = draw_anomaly_overlay(
                frame_bgr    = img,
                anomaly_info = result["anomaly_info"],
                alpha        = 0.4,
                object_mask=result.get("object_mask")
            )

        # 3. Label OK / NG + score góc trên-trái
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
                    (margin, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        return img

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _put_drop(q, item):
        """
        Đặt item vào queue; nếu full thì drop cũ rồi đặt mới.
        item = (frame_bgr, is_ok)  – is_ok: True | False | None
        """
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
    def _cleanup(self, thread_camera, thread_body, thread_sorting=None):
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

        if thread_sorting:
            thread_sorting.join(timeout=3)
            print("[Pipeline] SortingActuator đã dừng.")


        
        try:
            import pycuda.driver as cuda
            # Pop context nếu còn trong stack
            ctx = cuda.Context.get_current()
            if ctx is not None:
                ctx.pop()
        except Exception:
            pass


        print("[Pipeline] Stopped")

    # ----------------------------------------------------------------------- #
    def stop(self):
        print("[Pipeline] Stopping …")
        self.running = False