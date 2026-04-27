import json
import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from core.body_inspection import (
    U2NET_ENGINE_PATH,
    STD_H,
    STD_W,
    combine_liquid,
    device,
    dtype,
    draw_anomaly_overlay,
    draw_object_mask,
    merge_patchcore_with_masks,
    run_inspection_batch,
    _run_liquid_thread,
)
from core.path_manager import BASE_DIR

try:
    from torch2trt import TRTModule
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


TRT_ENGINE_PATH = os.path.join(BASE_DIR, "models/backbone/backbone_224x224.pth")
POSITION_BANK_FILENAMES = {
    "body": "body_memory_bank.pt",
    "cap": "cap_memory_bank.pt",
}


def load_system(name_project: str):
    engine_path = os.path.join(BASE_DIR, TRT_ENGINE_PATH)
    project_root = os.path.join(BASE_DIR, "projects", name_project)
    json_path = os.path.join(project_root, "project_info.json")

    model_engine = None
    banks = {}
    thresholds = {}

    if HAS_TRT and os.path.exists(engine_path):
        model_engine = TRTModule()
        model_engine.load_state_dict(torch.load(engine_path, map_location=device))
        model_engine.eval()

    for position, filename in POSITION_BANK_FILENAMES.items():
        bank_path = os.path.join(project_root, "memory_bank", filename)
        if os.path.exists(bank_path):
            banks[position] = torch.load(bank_path, map_location=device).to(dtype).contiguous()

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        settings = info.get("settings", {})
        thresholds.update(settings.get("thresholds", {}))
        if "body" not in thresholds and settings.get("threshold") is not None:
            thresholds["body"] = settings["threshold"]

    return model_engine, banks, thresholds


class BodyWorker(threading.Thread):
    def __init__(self, frame_input_queue, result_output_queue,
                 stop_event, infor_project, threshold: float = 18.0):
        super().__init__(daemon=True, name="BodyWorker")
        self.frame_input_queue = frame_input_queue
        self.result_output_queue = result_output_queue
        self.stop_event = stop_event
        self.default_threshold = threshold
        self.project_name = infor_project.get("project_name", "")
        self.project_root = Path(BASE_DIR) / "projects" / self.project_name

        self.model_engine, self.memory_banks, self.thresholds = load_system(self.project_name)
        if self.model_engine is None:
            raise RuntimeError("Khong load duoc model engine.")
        if not self.memory_banks:
            raise RuntimeError("Khong load duoc memory bank.")

        self._warmup()
        self._segmentor = None

    def _warmup(self, n: int = 3):
        dummy = torch.zeros((1, 3, STD_H, STD_W), device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(n):
                self.model_engine(dummy)
        torch.cuda.synchronize()

    def _init_segmentor(self):
        if not os.path.exists(U2NET_ENGINE_PATH):
            return
        try:
            from core.u2net_segmentor import U2NetSegmentor
            self._segmentor = U2NetSegmentor(U2NET_ENGINE_PATH)
            self._segmentor.attach()
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(2):
                self._segmentor.get_mask(dummy)
            self._segmentor.detach()
        except Exception:
            self._segmentor = None

    def _step1_get_masks(self, imgs_bgr: list) -> list:
        if self._segmentor is None:
            return [None] * len(imgs_bgr)
        try:
            self._segmentor.attach()
            masks = self._segmentor.get_masks_batch(imgs_bgr)
            self._segmentor.detach()
            return masks
        except Exception:
            try:
                self._segmentor.detach()
            except Exception:
                pass
            return [None] * len(imgs_bgr)

    @staticmethod
    def normalize_position(position: str) -> str:
        return str(position or "body").strip().lower()

    def _inspect_by_position(self, batch_images: list, masks: list, camera_positions: list[str]) -> list:
        results = [None] * len(batch_images)
        grouped_indices = {}
        for idx, position in enumerate(camera_positions):
            grouped_indices.setdefault(self.normalize_position(position), []).append(idx)

        for position, indices in grouped_indices.items():
            memory_bank = self.memory_banks.get(position)
            if memory_bank is None:
                continue
            threshold = float(self.thresholds.get(position, self.default_threshold))
            subset_images = [batch_images[i] for i in indices]
            subset_masks = [masks[i] for i in indices]
            patchcore_results = run_inspection_batch(subset_images, self.model_engine, memory_bank, threshold)
            merged = merge_patchcore_with_masks(patchcore_results, subset_masks, threshold)
            for local_idx, result in enumerate(merged):
                results[indices[local_idx]] = result
        return results

    def run(self):
        self._init_segmentor()
        while not self.stop_event.is_set():
            try:
                item = self.frame_input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self.frame_input_queue.task_done()
                break

            trigger_id, batch_images, camera_positions, display_keys = item

            try:
                imgs_bgr = [
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if img is not None and isinstance(img, np.ndarray) and img.ndim == 3
                    else None
                    for img in batch_images
                ]
                masks = self._step1_get_masks(imgs_bgr)

                liq_holder = [None]
                liq_done = threading.Event()
                threading.Thread(
                    target=_run_liquid_thread,
                    args=(imgs_bgr, masks, self.project_root, liq_holder, liq_done),
                    daemon=True,
                ).start()

                results = self._inspect_by_position(batch_images, masks, camera_positions)
                liq_done.wait(timeout=2.0)
                liquid_results = liq_holder[0] or [None] * len(batch_images)
                results = combine_liquid(results, liquid_results)

                for idx, result in enumerate(results):
                    if result is not None:
                        result["trigger_id"] = trigger_id
                        result["cam_id"] = idx
                        result["project_name"] = self.project_name
                        result["camera_position"] = camera_positions[idx]
                        result["display_key"] = display_keys[idx]

                self.result_output_queue.put((trigger_id, results))
            except Exception as e:
                self.result_output_queue.put((trigger_id, {"error": str(e)}))
            finally:
                self.frame_input_queue.task_done()
