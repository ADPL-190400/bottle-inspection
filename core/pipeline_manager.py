import threading
import queue
import cv2
import numpy as np
from hardware.camera.batch_camera import BatchCamera
from hardware.sorting.sorting_actuator import SortingActuator
from core.body_inspection import BodyWorker, draw_anomaly_overlay, draw_object_mask


class PipelineManager(threading.Thread):
    """
    Pipeline:
        BatchCamera → frames_queue
            ↓
        PipelineManager
            ├─ show_queue ← raw frame (is_ok=None)
            └─ body_in_queue → BodyWorker
                    [Bước 1] U2Net → masks
                    [Bước 2a] PatchCore  ┐ song song
                    [Bước 2b] Liquid     ┘
                    [Bước 3] merge + combine
                    → result_queue → _result_loop
                        ├─ show_queue ← overlay frame (is_ok)
                        └─ sorting_queue
    """

    def __init__(self, show_queue, infor_project: dict, threshold: float = 18,
                 batch_result_callback=None):
        super().__init__(daemon=True, name="PipelineManager")
        self.show_queue  = show_queue
        self.infor_project = infor_project
        self.threshold     = infor_project.get("settings", {}).get("threshold", threshold)
        self.running     = True

        # Callback (batch_ok: bool) → gọi 1 lần mỗi trigger (= 1 sản phẩm)
        self.batch_result_callback = batch_result_callback

        self.frames_queue  = queue.Queue(maxsize=1)
        self.body_in_queue = queue.Queue(maxsize=4)
        self.result_queue  = queue.Queue()
        self.sorting_queue = queue.Queue(maxsize=20)
        self._stop_event   = threading.Event()
        self._last_frames  = {}   # cam_key → BGR ndarray

    # ----------------------------------------------------------------------- #
    def run(self):
        thread_camera = thread_body = thread_sorting = None
        try:
            try:
                thread_camera = BatchCamera(self.frames_queue, self._stop_event, self.infor_project)
                thread_camera.start()
                print("[Pipeline] BatchCamera started.")
            except Exception as e:
                print(f"[Pipeline] ❌ BatchCamera: {e}")
                self.running = False
                return

            try:
                thread_sorting = SortingActuator(self.sorting_queue, self._stop_event)
                thread_sorting.start()
                print("[Pipeline] SortingActuator started.")
            except Exception as e:
                print(f"[Pipeline] ❌ SortingActuator: {e}")
                self.running = False
                return

            try:
                thread_body = BodyWorker(
                    frame_input_queue   = self.body_in_queue,
                    result_output_queue = self.result_queue,
                    stop_event          = self._stop_event,
                    threshold           = self.threshold,
                    infor_project       = self.infor_project,
                )
                thread_body.start()
                print("[Pipeline] BodyWorker started.")
            except Exception as e:
                print(f"[Pipeline] ❌ BodyWorker: {e}")
                self.running = False
                return

            threading.Thread(target=self._result_loop, daemon=True,
                             name="ResultReader").start()
            print("[Pipeline] Started")

            while self.running:
                try:
                    trigger_id, frames = self.frames_queue.get(timeout=1)
                except queue.Empty:
                    continue

                for cam_id, frame in enumerate(frames):
                    cam_key = str(cam_id + 1)
                    if frame is None:
                        continue
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._last_frames[cam_key] = frame_bgr
                    self._put_drop(self.show_queue.get(cam_key), (frame_bgr, None))

                self._put_drop(self.body_in_queue, (trigger_id, frames[:4]))

        except Exception as e:
            print(f"[Pipeline] ❌ {e}")
        finally:
            self._cleanup(thread_camera, thread_body, thread_sorting)

    # ----------------------------------------------------------------------- #
    def _result_loop(self):
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

            batch_ok = all(r["is_ok"] for r in results if r is not None)

            # ── Sorting ───────────────────────────────────────────────────── #
            try:
                self.sorting_queue.put_nowait(batch_ok)
            except queue.Full:
                try:
                    self.sorting_queue.get_nowait()
                except queue.Empty:
                    pass
                self.sorting_queue.put_nowait(batch_ok)

            # ── Cập nhật overlay lên từng cam ─────────────────────────────── #
            for cam_id, r in enumerate(results):
                if r is None:
                    continue
                cam_key = str(cam_id + 1)
                is_ok   = r["is_ok"]
                liq     = r.get("liquid_result")
                liq_str = (f"| liquid={'OK' if liq['is_ok'] else 'NG'} "
                           f"fill={liq['fill_ratio']:.1f}%") if liq else ""
                print(
                    f"[ResultReader] trigger={trigger_id} cam={cam_key} | "
                    f"{'✅ OK' if is_ok else '❌ NG'} | "
                    f"score={r['score']:.4f} | boxes={len(r['anomaly_info']['boxes'])} | "
                    f"{r['time_ms']:.1f}ms {liq_str}"
                )
                frame_bgr = self._last_frames.get(cam_key)
                if frame_bgr is not None:
                    overlay = self._draw_result(frame_bgr, r)
                    self._put_drop(self.show_queue.get(cam_key), (overlay, is_ok))

            print(f"[ResultReader] trigger={trigger_id} "
                  f"BATCH → {'✅ OK' if batch_ok else '❌ NG'}")

            # ── Callback 1 lần / trigger = 1 sản phẩm ────────────────────── #
            if self.batch_result_callback is not None:
                try:
                    self.batch_result_callback(batch_ok)
                except Exception as e:
                    print(f"[ResultReader] ⚠️ callback error: {e}")

            self.result_queue.task_done()

        print("[ResultReader] 🛑 Kết thúc.")

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _draw_result(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        """
        Vẽ theo thứ tự (không che nhau):
          1. Mask vùng vật thể — viền xanh lá + tô bán trong suốt
          2. Heatmap anomaly + defect boxes (chỉ khi NG)
          3. Liquid level lines + text
          4. Label OK/NG + score + fill ratio — góc trên-trái
        """
        object_mask = result.get("object_mask")

        # 1. Object mask
        img = draw_object_mask(frame_bgr, object_mask) if object_mask is not None \
            else frame_bgr.copy()

        # 2. Anomaly heatmap (chỉ khi NG, giới hạn trong mask)
        if not result["is_ok"] and result.get("anomaly_info"):
            img = draw_anomaly_overlay(img, result["anomaly_info"],
                                       alpha=0.4, object_mask=object_mask)

        # 3. Liquid level overlay
        liquid_result = result.get("liquid_result")
        if liquid_result is not None:
            try:
                from pathlib import Path
                from core.path_manager import BASE_DIR
                from core.liquid_level import LiquidLevelDetector
                project_name = result.get("project_name", "")
                project_root = Path(BASE_DIR) / "projects" / project_name
                detector     = LiquidLevelDetector(project_root)
                img          = detector.draw_on_existing(img, liquid_result)
            except Exception:
                pass

        # 4. Label + score + fill ratio — góc trên-trái
        is_ok  = result["is_ok"]
        color  = (0, 255, 0) if is_ok else (0, 0, 255)
        h, w   = img.shape[:2]
        margin = 20

        panel_h = 110 if liquid_result is not None else 90
        bg = img.copy()
        cv2.rectangle(bg, (0, 0), (w // 3, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.45, img, 0.55, 0, img)

        cv2.putText(img, "OK" if is_ok else "NG",
                    (margin, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
        cv2.putText(img, f"score: {result['score']:.3f}",
                    (margin, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        if liquid_result is not None:
            liq_color = (0, 255, 0) if liquid_result["is_ok"] else (0, 100, 255)
            cv2.putText(
                img,
                f"fill: {liquid_result['fill_ratio']:.1f}%  "
                f"[{liquid_result['min_fill']:.0f}-{liquid_result['max_fill']:.0f}]",
                (margin, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, liq_color, 1, cv2.LINE_AA,
            )

        return img

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _put_drop(q, item):
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
        if thread_body:
            thread_body.join(timeout=5)
        if thread_sorting:
            thread_sorting.join(timeout=3)
        print("[Pipeline] Stopped")

    def stop(self):
        print("[Pipeline] Stopping …")
        self.running = False