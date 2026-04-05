import threading
import queue
import os
import time
import torch
import numpy as np
from pathlib import Path
from core.path_manager import BASE_DIR
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2

try:
    from torch2trt import TRTModule
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

U2NET_ENGINE_PATH = os.path.join(BASE_DIR, "models/remove_bg/u2netp.trt")
BANK_CACHE_PATH   = "memory_bank/memory_bank.pt"
TRT_ENGINE_PATH   = "models/backbone/backbone_224x224.pth"
STD_W, STD_H      = 224, 224
FEAT_H = FEAT_W   = 28

device   = torch.device("cuda")
dtype    = torch.float16
MEAN     = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD_NORM = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)


# =========================================================================== #
#  TIỀN XỬ LÝ                                                                 #
# =========================================================================== #
def _to_pil_rgb(img_input) -> tuple:
    if isinstance(img_input, (str, Path)):
        from PIL import Image as _Image
        img = _Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        from PIL import Image as _Image
        arr = np.ascontiguousarray(img_input)
        if arr.ndim == 2:
            img = _Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            c = arr.shape[2]
            if c == 3:
                img = _Image.fromarray(arr, mode="RGB")
            elif c == 4:
                img = _Image.fromarray(arr, mode="RGBA").convert("RGB")
            else:
                img = _Image.fromarray(arr).convert("RGB")
        else:
            img = _Image.fromarray(arr).convert("RGB")
    else:
        img = img_input.convert("RGB")

    orig_size = img.size
    from PIL import Image as _Image
    img_res  = img.resize((STD_W, STD_H), _Image.LANCZOS)
    arr_out  = np.array(img_res, dtype=np.uint8)
    return arr_out, orig_size


def _arr_to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = (
        torch.from_numpy(arr)
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=dtype)
    )
    return (t / 255.0 - MEAN) / STD_NORM


# =========================================================================== #
#  ANOMALY MAP                                                                 #
# =========================================================================== #
def _build_anomaly_map(patch_scores, orig_w, orig_h, threshold,
                       object_mask: np.ndarray = None) -> dict:
    if isinstance(patch_scores, torch.Tensor):
        scores_2d = patch_scores.cpu().float().numpy().reshape(FEAT_H, FEAT_W)
    else:
        scores_2d = np.asarray(patch_scores, dtype=np.float32).reshape(FEAT_H, FEAT_W)

    if object_mask is not None:
        mask_feat = cv2.resize(object_mask, (FEAT_W, FEAT_H),
                               interpolation=cv2.INTER_NEAREST)
        scores_2d = scores_2d * (mask_feat > 127).astype(np.float32)

    heatmap_full = cv2.resize(scores_2d, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)
    s_min, s_max = heatmap_full.min(), heatmap_full.max()
    heatmap_norm = (
        ((heatmap_full - s_min) / (s_max - s_min) * 255).astype(np.uint8)
        if s_max > s_min else np.zeros((orig_h, orig_w), dtype=np.uint8)
    )
    anomaly_mask = (heatmap_full > threshold).astype(np.uint8)
    boxes = []
    if anomaly_mask.any():
        contours, _ = cv2.findContours(
            anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((x, y, x + bw, y + bh))

    return {
        "heatmap_gray"   : heatmap_norm,
        "anomaly_mask"   : anomaly_mask.astype(bool),
        "boxes"          : boxes,
        "patch_scores_2d": scores_2d,
    }


def draw_anomaly_overlay(frame_bgr: np.ndarray, anomaly_info: dict,
                         alpha: float = 0.5,
                         object_mask: np.ndarray = None) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    heatmap_color = cv2.applyColorMap(anomaly_info["heatmap_gray"], cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))
    blended       = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    if object_mask is not None:
        mask_r = cv2.resize(object_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        blended[mask_r <= 127] = img[mask_r <= 127]
    for (x1, y1, x2, y2) in anomaly_info["boxes"]:
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(blended, "DEFECT", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return blended


def draw_object_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return frame_bgr
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary        = (mask > 127).astype(np.uint8)
    overlay       = img.copy()
    obj_region    = binary.astype(bool)
    overlay[obj_region] = (
        overlay[obj_region].astype(np.float32) * 0.6
        + np.array([0, 180, 0], np.float32) * 0.4
    ).astype(np.uint8)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 220, 0), 2)
    return img


# =========================================================================== #
#  PATCHCORE INFERENCE                                                         #
# =========================================================================== #
def run_inspection_batch(images: list, model_engine, memory_bank,
                         threshold: float = 18.0) -> list:
    """Chạy PatchCore, lưu patch_scores_raw để merge có thể tính lại trong mask."""
    print("[BodyInspection] run inspection")
    results = [None] * len(images)
    valid   = [(i, img) for i, img in enumerate(images) if img is not None]
    if not valid:
        return results

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=len(valid)) as pool:
        preprocessed = list(pool.map(lambda a: (a[0], *_to_pil_rgb(a[1])), valid))
    preprocessed.sort(key=lambda x: x[0])

    features_list = []
    torch.cuda.synchronize()
    with torch.inference_mode():
        for cam_id, arr, orig_size in preprocessed:
            feat = model_engine(_arr_to_tensor(arr))
            features_list.append((cam_id, orig_size, feat))

    with torch.inference_mode():
        K            = len(features_list)
        feat_f32     = torch.cat([f for _, _, f in features_list], dim=0).float()
        bank_f32     = memory_bank.float()
        a_sq         = (feat_f32 ** 2).sum(dim=1, keepdim=True)
        b_sq         = (bank_f32 ** 2).sum(dim=1).unsqueeze(0)
        dist_sq      = (a_sq + b_sq - 2.0 * torch.mm(feat_f32, bank_f32.t())).clamp(min=1e-6)
        patch_dist   = dist_sq.min(dim=1).values.sqrt().reshape(K, FEAT_H * FEAT_W)
        scores       = patch_dist.max(dim=1).values

    torch.cuda.synchronize()
    ms_each = (time.perf_counter() - t0) * 1000 / K

    for k, (cam_id, (w, h), _) in enumerate(features_list):
        s            = scores[k].item()
        raw          = patch_dist[k].cpu().float().numpy()   # (784,)
        anomaly_info = _build_anomaly_map(raw, w, h, threshold, object_mask=None)
        anomaly_info["patch_scores_raw"] = raw
        results[cam_id] = {
            "score"       : s,
            "is_ok"       : s <= threshold,
            "result"      : "NG (LỖI)" if s > threshold else "OK (ĐẠT)",
            "time_ms"     : ms_each,
            "img_size"    : (w, h),
            "anomaly_info": anomaly_info,
            "liquid_result": None,
            "object_mask" : None,
        }

    print(f"--- RAW ({K} ảnh | ~{ms_each:.1f}ms/ảnh) ---")
    for cam_id, r in enumerate(results):
        if r:
            print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}")
    return results


# =========================================================================== #
#  MERGE: PATCHCORE + U2NET MASK                                               #
# =========================================================================== #
def merge_patchcore_with_masks(patchcore_results: list,
                               masks: list,
                               threshold: float) -> list:
    """Tính lại score chỉ trong vùng mask, rebuild anomaly_map."""
    merged = []
    for cam_id, (result, mask) in enumerate(zip(patchcore_results, masks)):
        if result is None:
            merged.append(None)
            continue

        r = result.copy()
        r["anomaly_info"] = result["anomaly_info"].copy()
        r["object_mask"]  = mask

        if mask is not None:
            w, h             = r["img_size"]
            mask_feat        = cv2.resize(mask, (FEAT_W, FEAT_H),
                                          interpolation=cv2.INTER_NEAREST)
            raw              = r["anomaly_info"]["patch_scores_raw"]
            scores_in_mask   = raw.reshape(FEAT_H, FEAT_W)[mask_feat > 127]
            score_in_mask    = float(scores_in_mask.max()) if scores_in_mask.size > 0 else 0.0

            if scores_in_mask.size == 0:
                print(f"[Merge] cam{cam_id+1} — mask quá nhỏ → score=0 → OK")

            is_ok        = score_in_mask <= threshold
            anomaly_info = _build_anomaly_map(raw, w, h, threshold, object_mask=mask)
            anomaly_info["patch_scores_raw"] = raw

            r["score"]        = score_in_mask
            r["is_ok"]        = is_ok
            r["result"]       = "OK (ĐẠT)" if is_ok else "NG (LỖI)"
            r["anomaly_info"] = anomaly_info

            print(f"[Merge] cam{cam_id+1} — "
                  f"score_full={result['score']:.4f} → "
                  f"score_mask={score_in_mask:.4f} → {r['result']}")

        merged.append(r)
    return merged


# =========================================================================== #
#  LIQUID CHECK (chạy trong thread riêng)                                      #
# =========================================================================== #
def _run_liquid_thread(imgs_bgr: list, masks: list,
                       project_root: Path,
                       holder: list, done: threading.Event):
    """
    Thread target cho liquid check.
    Dùng masks có sẵn từ U2Net — KHÔNG inference thêm.

    Đọc từ liquid_config.json:
        enabled  : bool       — bật/tắt toàn bộ liquid check
        cam_ids  : list[int]  — 0-indexed, chỉ check các cam này

    holder[0] = list[dict|None]
    """
    liquid_results = [None] * len(imgs_bgr)
    try:
        from core.liquid_level import LiquidLevelDetector
        detector = LiquidLevelDetector(project_root)
        if not detector.is_ready():
            return   # chưa setup → giữ None

        config, _ = detector.load()

        # ── Kiểm tra bật/tắt ─────────────────────────────────────────
        if not config.get("enabled", True):
            print("[LiquidThread] ℹ️  Liquid check đang tắt — bỏ qua")
            return

        # ── Danh sách cam cần check (0-indexed) ──────────────────────
        # Nếu không có cam_ids trong config → check tất cả (backward compat)
        cam_ids_to_check = config.get("cam_ids", list(range(len(imgs_bgr))))

        print(f"[LiquidThread] Check liquid cam: {[c+1 for c in cam_ids_to_check]}")

        for cam_id, (img_bgr, mask) in enumerate(zip(imgs_bgr, masks)):
            if cam_id not in cam_ids_to_check:
                continue   # ← skip cam không cần check → tiết kiệm ~200ms/cam
            if img_bgr is None:
                continue
            try:
                liq = detector.detect(img_bgr, object_mask=mask)
                liquid_results[cam_id] = liq
                if liq:
                    label = "✅ OK" if liq["is_ok"] else "❌ NG"
                    print(f"[LiquidThread] cam{cam_id+1} {label}  "
                          f"fill={liq['fill_ratio']:.1f}%  "
                          f"[{liq['min_fill']:.1f},{liq['max_fill']:.1f}]")
            except Exception as e:
                print(f"[LiquidThread] ⚠️  cam{cam_id+1}: {e}")
    except Exception as e:
        print(f"[LiquidThread] ⚠️  {e}")
    finally:
        holder[0] = liquid_results
        done.set()


def combine_liquid(merged_results: list, liquid_results: list) -> list:
    """
    Gắn liquid_result vào từng cam result.

    Logic liquid: ÍT NHẤT 1 cam liquid OK → coi như liquid OK cho cả batch.
    Kết quả cuối mỗi cam = body_ok AND liquid_batch_ok.
    """
    # ── Tính liquid_batch_ok: ít nhất 1 cam có liquid OK ────────────────
    liquid_batch_ok = any(
        liq["is_ok"]
        for liq in liquid_results
        if liq is not None
    )
    has_any_liquid = any(liq is not None for liq in liquid_results)

    if has_any_liquid:
        label = "✅ OK" if liquid_batch_ok else "❌ NG"
        ok_cams = [i+1 for i, liq in enumerate(liquid_results)
                   if liq is not None and liq["is_ok"]]
        print(f"[Combine] Liquid batch: {label}  "
              f"(cam OK: {ok_cams if ok_cams else 'none'})")

    # ── Gắn vào từng cam result ──────────────────────────────────────────
    for cam_id, (r, liq) in enumerate(zip(merged_results, liquid_results)):
        if r is None:
            continue
        r["liquid_result"]       = liq
        r["liquid_batch_ok"]     = liquid_batch_ok if has_any_liquid else None

        if has_any_liquid:
            combined_ok  = r["is_ok"] and liquid_batch_ok
            r["is_ok"]   = combined_ok
            r["result"]  = "OK (ĐẠT)" if combined_ok else "NG (LỖI)"
            liq_str      = f"fill={liq['fill_ratio']:.1f}%" if liq else "N/A"
            print(f"[Combine] cam{cam_id+1} "
                  f"body={'OK' if r['is_ok'] else 'NG'} "
                  f"liquid_cam={liq_str} "
                  f"liquid_batch={'OK' if liquid_batch_ok else 'NG'} "
                  f"→ {r['result']}")

    return merged_results


# =========================================================================== #
#  LOAD HỆ THỐNG                                                               #
# =========================================================================== #
def load_system(name_project):
    engine_path = os.path.join(BASE_DIR, TRT_ENGINE_PATH)
    bank_path   = os.path.join(BASE_DIR, "projects", name_project, BANK_CACHE_PATH)

    model_engine = None
    bank         = None

    if HAS_TRT and os.path.exists(engine_path):
        print(f"🚀 Loading TRT Engine: {engine_path}")
        model_engine = TRTModule()
        model_engine.load_state_dict(torch.load(engine_path, map_location=device))
        model_engine.eval()
    else:
        print("⚠️  Không tìm thấy TRT engine.")

    if os.path.exists(bank_path):
        print(f"📂 Loading Memory Bank: {bank_path}")
        bank = torch.load(bank_path, map_location=device).to(dtype).contiguous()
        print(f"   Bank shape: {bank.shape}")
    else:
        print("⚠️  Memory Bank chưa có.")

    return model_engine, bank


# =========================================================================== #
#  BODY WORKER                                                                 #
# =========================================================================== #
class BodyWorker(threading.Thread):
    """
    Luồng xử lý mỗi trigger — 3 bước tuần tự/song song:

        [Bước 1] U2Net (sequential, ~30ms)
                    → masks[4]
                    │
                    ├─ [Bước 2a] PatchCore          ─────────────┐ SONG SONG
                    └─ [Bước 2b] LiquidCheck(masks) ─────────────┘
                    │
                    └─ [Bước 3] merge + combine → result_queue
    """

    def __init__(self, frame_input_queue, result_output_queue,
                 stop_event, infor_project, threshold: float = 18.0):
        super().__init__(daemon=True, name="BodyWorker")
        self.frame_input_queue   = frame_input_queue
        self.result_output_queue = result_output_queue
        self.stop_event          = stop_event
        self.threshold           = threshold
        self.project_name        = infor_project.get("project_name", "")
        self.project_root        = Path(BASE_DIR) / "projects" / self.project_name

        self.model_engine, self.memory_bank = load_system(self.project_name)
        if self.model_engine is None:
            raise RuntimeError("❌ Không load được model engine.")
        if self.memory_bank is None:
            raise RuntimeError("❌ Không load được memory bank.")

        self._warmup()
        self._segmentor = None   # init trong run() đúng pycuda context

    # ------------------------------------------------------------------ #
    def _warmup(self, n: int = 3):
        print(f"[{self.name}] 🔥 Warming up ({n} lần)...")
        dummy = torch.zeros((1, 3, STD_H, STD_W), device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(n):
                self.model_engine(dummy)
        torch.cuda.synchronize()
        print(f"[{self.name}] ✅ Warm-up xong.")

    # ------------------------------------------------------------------ #
    def _init_segmentor(self):
        if not os.path.exists(U2NET_ENGINE_PATH):
            print(f"[{self.name}] ⚠️  U2Net engine không tìm thấy.")
            return
        try:
            from core.u2net_segmentor import U2NetSegmentor
            self._segmentor = U2NetSegmentor(U2NET_ENGINE_PATH)
            self._segmentor.attach()
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(2):
                self._segmentor.get_mask(dummy)
            self._segmentor.detach()
            print(f"[{self.name}] ✅ U2Net ready.")
        except Exception as e:
            print(f"[{self.name}] ⚠️  U2Net load thất bại: {e}")
            self._segmentor = None

    # ------------------------------------------------------------------ #
    def _step1_get_masks(self, imgs_bgr: list) -> list:
        """Bước 1: U2Net sequential, trả masks trước khi fork."""
        if self._segmentor is None:
            return [None] * len(imgs_bgr)
        try:
            self._segmentor.attach()
            masks = self._segmentor.get_masks_batch(imgs_bgr)
            self._segmentor.detach()
            return masks
        except Exception as e:
            print(f"[{self.name}] ⚠️  U2Net lỗi: {e}")
            try:
                self._segmentor.detach()
            except Exception:
                pass
            return [None] * len(imgs_bgr)

    # ------------------------------------------------------------------ #
    def run(self):
        self._init_segmentor()
        print(f"[{self.name}] ▶ Bắt đầu …")

        while not self.stop_event.is_set():
            try:
                item = self.frame_input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                print(f"[{self.name}] 🛑 Nhận lệnh dừng.")
                self.frame_input_queue.task_done()
                break

            trigger_id, batch_images = item

            try:
                t_start = time.perf_counter()

                # Chuẩn bị BGR (camera gửi RGB)
                imgs_bgr = [
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if img is not None and isinstance(img, np.ndarray) and img.ndim == 3
                    else None
                    for img in batch_images
                ]

                # ── Bước 1: U2Net trước ───────────────────────────────
                t_u2 = time.perf_counter()
                masks = self._step1_get_masks(imgs_bgr)
                print(f"[{self.name}] U2Net done: {(time.perf_counter()-t_u2)*1000:.1f}ms")

                # ── Bước 2: PatchCore + Liquid SONG SONG ─────────────
                liq_holder = [None]
                liq_done   = threading.Event()

                # 2b: Liquid thread (dùng masks vừa có)
                threading.Thread(
                    target=_run_liquid_thread,
                    args=(imgs_bgr, masks, self.project_root, liq_holder, liq_done),
                    daemon=True,
                    name=f"LiquidThread-{trigger_id}",
                ).start()

                # 2a: PatchCore (chạy ngay, không đợi liquid)
                patchcore_results = run_inspection_batch(
                    batch_images, self.model_engine, self.memory_bank, self.threshold
                )

                # ── Bước 3: merge + combine ───────────────────────────
                merged = merge_patchcore_with_masks(
                    patchcore_results, masks, self.threshold
                )

                # Đợi liquid (thường xong trước PatchCore)
                liq_done.wait(timeout=2.0)
                if not liq_done.is_set():
                    print(f"[{self.name}] ⚠️  Liquid timeout trigger={trigger_id}")

                liquid_results = liq_holder[0] or [None] * len(batch_images)
                results        = combine_liquid(merged, liquid_results)

                # Metadata
                for cam_id, r in enumerate(results):
                    if r is not None:
                        r["trigger_id"]   = trigger_id
                        r["cam_id"]       = cam_id
                        r["project_name"] = self.project_name

                t_total = (time.perf_counter() - t_start) * 1000
                print(f"\n--- KẾT QUẢ CUỐI trigger={trigger_id} | {t_total:.1f}ms ---")
                for cam_id, r in enumerate(results):
                    if r is None:
                        print(f"  cam[{cam_id}] ⚫ SKIP")
                    else:
                        liq = r.get("liquid_result")
                        liq_s = f"fill={liq['fill_ratio']:.1f}%" if liq else "liquid=N/A"
                        print(f"  cam[{cam_id}] {r['result']}  "
                              f"score={r['score']:.4f}  "
                              f"mask={'✓' if r['object_mask'] is not None else '✗'}  "
                              f"{liq_s}")

                self.result_output_queue.put((trigger_id, results))

            except Exception as e:
                print(f"[{self.name}] ⚠️  Lỗi trigger {trigger_id}: {e}")
                self.result_output_queue.put((trigger_id, {"error": str(e)}))
            finally:
                self.frame_input_queue.task_done()

        print(f"[{self.name}] ✅ Kết thúc.")