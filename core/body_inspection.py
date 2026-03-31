import threading
import queue
import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2
import json

try:
    from torch2trt import torch2trt, TRTModule
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

U2NET_ENGINE_PATH = "models/remove_bg/u2netp.trt"
BANK_CACHE_PATH   = "memory_bank.pt"
TRT_ENGINE_PATH   = "backbone_flex_224x224.pth"
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
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        arr = np.ascontiguousarray(img_input)
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            c = arr.shape[2]
            if c == 3:
                img = Image.fromarray(arr, mode="RGB")
            elif c == 4:
                img = Image.fromarray(arr, mode="RGBA").convert("RGB")
            else:
                img = Image.fromarray(arr).convert("RGB")
        else:
            img = Image.fromarray(arr).convert("RGB")
    else:
        img = img_input.convert("RGB")

    orig_size = img.size          # (W, H)
    img_res   = img.resize((STD_W, STD_H), Image.LANCZOS)
    arr_out   = np.array(img_res, dtype=np.uint8)
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
                       object_mask: np.ndarray = None):
    """
    Xây dựng anomaly map từ patch scores.

    Args:
        patch_scores : torch.Tensor hoặc np.ndarray, shape (FEAT_H*FEAT_W,)
        orig_w, orig_h : kích thước ảnh gốc
        threshold    : ngưỡng anomaly
        object_mask  : uint8 [0,255] shape (H,W) từ U2Net — nếu có, zero-out
                       các patch nằm ngoài vùng vật thể trước khi build map.
    """
    # ── chuyển sang numpy 2D ──────────────────────────────────────────────
    if isinstance(patch_scores, torch.Tensor):
        scores_2d = patch_scores.cpu().float().numpy().reshape(FEAT_H, FEAT_W)
    else:
        scores_2d = np.asarray(patch_scores, dtype=np.float32).reshape(FEAT_H, FEAT_W)

    # ── zero-out patch ngoài mask (nếu có) ───────────────────────────────
    if object_mask is not None:
        mask_feat = cv2.resize(object_mask, (FEAT_W, FEAT_H),
                               interpolation=cv2.INTER_NEAREST)
        scores_2d = scores_2d * (mask_feat > 127).astype(np.float32)

    # ── resize về kích thước gốc ─────────────────────────────────────────
    heatmap_full = cv2.resize(scores_2d, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)

    s_min, s_max = heatmap_full.min(), heatmap_full.max()
    if s_max > s_min:
        heatmap_norm = ((heatmap_full - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        heatmap_norm = np.zeros((orig_h, orig_w), dtype=np.uint8)

    anomaly_mask = (heatmap_full > threshold).astype(np.uint8)

    boxes = []
    if anomaly_mask.any():
        contours, _ = cv2.findContours(
            anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((x, y, x + bw, y + bh))

    return {
        "heatmap_gray"   : heatmap_norm,
        "anomaly_mask"   : anomaly_mask.astype(bool),
        "boxes"          : boxes,
        "patch_scores_2d": scores_2d,   # đã được masked (ngoài mask = 0)
    }


def draw_anomaly_overlay(frame_bgr: np.ndarray, anomaly_info: dict,
                         alpha: float = 0.5,
                         object_mask: np.ndarray = None) -> np.ndarray:
    """
    Vẽ heatmap lên frame.
    Nếu object_mask được truyền vào, heatmap CHỈ hiển thị trong vùng mask;
    vùng ngoài mask giữ nguyên pixel gốc.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]


    heatmap_color = cv2.applyColorMap(anomaly_info["heatmap_gray"], cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))



    blended = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)

    # Giới hạn heatmap trong vùng object
    if object_mask is not None:
        mask_resized = cv2.resize(object_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool    = mask_resized > 127
        blended[~mask_bool] = img[~mask_bool]

    for (x1, y1, x2, y2) in anomaly_info["boxes"]:
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(blended, "DEFECT", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        

    return blended


# =========================================================================== #
#  MASK UTILITIES                                                              #
# =========================================================================== #
def draw_object_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Vẽ vùng vật thể từ U2Net lên frame:
      - Tô màu xanh lá bán trong suốt lên vùng object
      - Viền xanh lá đậm bao quanh contour
    """
    if mask is None:
        return frame_bgr

    img = frame_bgr.copy()
    h, w = img.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > 127).astype(np.uint8)

    overlay       = img.copy()
    object_region = binary.astype(bool)
    overlay[object_region] = (
        overlay[object_region].astype(np.float32) * 0.6
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
    """
    Chạy PatchCore inference cho batch ảnh.
    Trả về list result với patch_scores_2d thô (chưa apply mask).
    Mask sẽ được apply ở bước merge_patchcore_with_masks.
    """
    print("[BodyInspection] run inspection")
    results = [None] * len(images)

    valid = [(cam_id, img) for cam_id, img in enumerate(images) if img is not None]
    if not valid:
        print("⚠️  Batch toàn None – bỏ qua inference.")
        return results

    t0 = time.perf_counter()

    def preprocess_one(args):
        cam_id, img = args
        arr, orig_size = _to_pil_rgb(img)
        return cam_id, arr, orig_size

    with ThreadPoolExecutor(max_workers=len(valid)) as pool:
        preprocessed = list(pool.map(preprocess_one, valid))
    preprocessed.sort(key=lambda x: x[0])

    features_list = []
    torch.cuda.synchronize()
    with torch.inference_mode():
        for cam_id, arr, orig_size in preprocessed:
            img_t = _arr_to_tensor(arr)
            feat  = model_engine(img_t)
            features_list.append((cam_id, orig_size, feat))

    with torch.inference_mode():
        features_all    = torch.cat([f for _, _, f in features_list], dim=0)
        K               = len(features_list)

        feat_f32 = features_all.float()
        bank_f32 = memory_bank.float()

        a_sq    = (feat_f32 ** 2).sum(dim=1, keepdim=True)
        b_sq    = (bank_f32 ** 2).sum(dim=1).unsqueeze(0)
        dist_sq = (a_sq + b_sq - 2.0 * torch.mm(feat_f32, bank_f32.t())).clamp(min=1e-6)

        patch_dist = dist_sq.min(dim=1).values.sqrt().reshape(K, FEAT_H * FEAT_W)
        scores     = patch_dist.max(dim=1).values

    torch.cuda.synchronize()
    ms_total = (time.perf_counter() - t0) * 1000
    ms_each  = ms_total / K

    for k, (cam_id, (w, h), _) in enumerate(features_list):
        s     = scores[k].item()
        # Build anomaly map KHÔNG có mask ở đây — mask sẽ apply ở merge
        anomaly_info = _build_anomaly_map(patch_dist[k], w, h, threshold,
                                          object_mask=None)
        # Lưu raw patch_scores để merge có thể tính lại score trong mask
        anomaly_info["patch_scores_raw"] = patch_dist[k].cpu().float().numpy()

        label = "NG (LỖI)" if s > threshold else "OK (ĐẠT)"
        results[cam_id] = {
            "score"       : s,
            "is_ok"       : s <= threshold,
            "result"      : label,
            "time_ms"     : ms_each,
            "img_size"    : (w, h),
            "anomaly_info": anomaly_info,
        }

    print(f"\n--- KẾT QUẢ RAW ({K}/{len(images)} ảnh | "
          f"tổng {ms_total:.1f}ms | mỗi ảnh ~{ms_each:.1f}ms) ---")
    for cam_id, r in enumerate(results):
        if r is None:
            print(f"  cam[{cam_id}] ⚫ SKIP (None)")
        else:
            w, h  = r["img_size"]
            print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}  {w}x{h}")

    return results


# =========================================================================== #
#  MERGE: PATCHCORE + U2NET MASK                                               #
#  Toàn bộ logic OK/NG được quyết định tại đây dựa trên score TRONG mask.     #
# =========================================================================== #
def merge_patchcore_with_masks(patchcore_results: list,
                               masks: list,
                               threshold: float) -> list:
    """
    Merge PatchCore results + U2Net masks.

    Logic chính:
      1. Resize mask về feature map (28×28)
      2. Chỉ lấy patch scores nằm TRONG mask
      3. score_in_mask = max(scores trong mask)  →  so với threshold → OK/NG
      4. Rebuild anomaly_map với mask (heatmap + boxes chỉ trong mask)
      5. Gắn object_mask vào result để UI vẽ overlay
    """
    merged = []

    for cam_id, (result, mask) in enumerate(zip(patchcore_results, masks)):
        if result is None:
            merged.append(None)
            continue

        r = result.copy()
        r["anomaly_info"] = result["anomaly_info"].copy()
        r["object_mask"]  = mask

        if mask is not None:
            # cv2.imwrite('debug_mask.png',mask)
            w, h = r["img_size"]

            # ── Resize mask → feature map size ───────────────────────────
            mask_feat      = cv2.resize(mask, (FEAT_W, FEAT_H),
                                        interpolation=cv2.INTER_NEAREST)
            mask_feat_bool = mask_feat > 127

            # ── Lấy raw patch scores (flat, 784,) ─────────────────────────
            patch_scores_raw = r["anomaly_info"]["patch_scores_raw"]   # (784,)
            scores_2d_raw    = patch_scores_raw.reshape(FEAT_H, FEAT_W)

            # ── Tính score CHỈ trong mask ─────────────────────────────────
            scores_in_mask = scores_2d_raw[mask_feat_bool]

            if scores_in_mask.size > 0:
                score_in_mask = float(scores_in_mask.max())
            else:
                # Không có patch nào trong mask → coi như OK
                score_in_mask = 100000
                print(f"[Merge] cam{cam_id + 1} — mask quá nhỏ, không có patch → OK")

            is_ok = score_in_mask <= threshold
            label = "OK (ĐẠT)" if is_ok else "NG (LỖI)"

            # ── Rebuild anomaly map với mask ──────────────────────────────
            anomaly_info = _build_anomaly_map(
                patch_scores_raw,   # numpy array (784,)
                w, h, threshold,
                object_mask=mask,   # zero-out ngoài mask
            )
            # Giữ lại raw scores để debug nếu cần
            anomaly_info["patch_scores_raw"] = patch_scores_raw

            r["score"]        = score_in_mask
            r["is_ok"]        = is_ok
            r["result"]       = label
            r["anomaly_info"] = anomaly_info

            print(
                f"[Merge] cam{cam_id + 1} — "
                f"score_full={result['score']:.4f} | "
                f"score_in_mask={score_in_mask:.4f} | "
                f"threshold={threshold} | "
                f"{result['result']} → {label}"
            )

        else:
            # Không có mask → giữ nguyên result gốc, chỉ gắn thêm object_mask=None
            pass

        merged.append(r)

    return merged


# =========================================================================== #
#  LOAD HỆ THỐNG                                                               #
# =========================================================================== #
# def load_threshold_from_json(project_root: Path) -> float | None:
#     """Đọc threshold từ project_info.json. Trả về None nếu chưa có."""
#     json_path = project_root / PROJECT_INFO_FILENAME
#     if not json_path.exists():
#         return None
#     with open(json_path, "r", encoding="utf-8") as f:
#         info = json.load(f)
#     return info.get("settings", {}).get("threshold", None)

def load_system():
    model_engine = None
    bank         = None

    if HAS_TRT and os.path.exists(TRT_ENGINE_PATH):
        print(f"🚀 Loading TRT Engine: {TRT_ENGINE_PATH}")
        model_engine = TRTModule()
        model_engine.load_state_dict(torch.load(TRT_ENGINE_PATH, map_location=device))
        model_engine.eval()
    else:
        print("⚠️  Không tìm thấy TRT engine.")

    if os.path.exists(BANK_CACHE_PATH):
        print(f"📂 Loading Memory Bank: {BANK_CACHE_PATH}")
        bank = torch.load(BANK_CACHE_PATH, map_location=device).to(dtype).contiguous()
        print(f"   Bank shape: {bank.shape}")
    else:
        print("⚠️  Memory Bank chưa có.")

    return model_engine, bank


# =========================================================================== #
#  BODY WORKER                                                                 #
# =========================================================================== #
class BodyWorker(threading.Thread):
    """
    U2Net chạy SONG SONG với PatchCore trong thread riêng.

        frame_input_queue → (trigger_id, [img1..img4])
                │
                ├─ U2NetThread  ── attach() → get_masks_batch() → detach() → masks[4]
                │                                                                  │
                └─ BodyWorker   ── run_inspection_batch() → patchcore_results[4]  │
                                                                                   ▼
                                          merge_patchcore_with_masks() → result_queue

    Sau merge: score, is_ok, result đều dựa trên score TRONG mask U2Net.
    """

    def __init__(self, frame_input_queue, result_output_queue,
                 stop_event, threshold: float = 18.0):
        super().__init__(daemon=True, name="BodyWorker")
        self.frame_input_queue   = frame_input_queue
        self.result_output_queue = result_output_queue
        self.stop_event          = stop_event
        self.threshold           = threshold

        self.model_engine, self.memory_bank = load_system()
        if self.model_engine is None:
            raise RuntimeError("❌ Không load được model engine.")
        if self.memory_bank is None:
            raise RuntimeError("❌ Không load được memory bank.")

        self._warmup()
        self._segmentor = None   # khởi tạo trong run()

    # ------------------------------------------------------------------ #
    def _warmup(self, n: int = 3):
        print(f"[{self.name}] 🔥 Warming up TRT engine ({n} lần)...")
        dummy = torch.zeros((1, 3, STD_H, STD_W), device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(n):
                _ = self.model_engine(dummy)
        torch.cuda.synchronize()
        print(f"[{self.name}] ✅ Warm-up PatchCore xong.")

    # ------------------------------------------------------------------ #
    def _init_segmentor(self):
        """Khởi tạo U2Net trong run() thread."""
        if not os.path.exists(U2NET_ENGINE_PATH):
            print(f"[{self.name}] ⚠️  Không tìm thấy {U2NET_ENGINE_PATH} — chạy không có mask")
            return
        try:
            from core.u2net_segmentor import U2NetSegmentor
            self._segmentor = U2NetSegmentor(U2NET_ENGINE_PATH)

            self._segmentor.attach()
            dummy_bgr = np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(2):
                self._segmentor.get_mask(dummy_bgr)
            self._segmentor.detach()

            print(f"[{self.name}] ✅ U2Net ready.")
        except Exception as e:
            print(f"[{self.name}] ⚠️  U2Net load thất bại: {e} — chạy không có mask")
            self._segmentor = None

    # ------------------------------------------------------------------ #
    def _run_u2net_parallel(self, imgs_bgr: list,
                            result_holder: list,
                            done_event: threading.Event):
        """Chạy trong daemon thread riêng, song song với PatchCore."""
        try:
            if self._segmentor is not None:
                self._segmentor.attach()
                masks = self._segmentor.get_masks_batch(imgs_bgr)
                self._segmentor.detach()
            else:
                masks = [None] * len(imgs_bgr)
        except Exception as e:
            print(f"[U2NetThread] ⚠️  Lỗi segmentation: {e}")
            try:
                self._segmentor.detach()
            except Exception:
                pass
            masks = [None] * len(imgs_bgr)
        finally:
            result_holder[0] = masks
            done_event.set()

    # ------------------------------------------------------------------ #
    def run(self):
        self._init_segmentor()
        print(f"[{self.name}] ▶ Bắt đầu. Chờ từ queue …")

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

                # Chuẩn bị BGR cho U2Net (camera gửi RGB)
                imgs_bgr = []
                for img in batch_images:
                    if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                        imgs_bgr.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        imgs_bgr.append(None)

                # ── U2Net thread (song song với PatchCore) ─────────────
                masks_holder = [None]
                u2net_done   = threading.Event()
                u2net_thread = threading.Thread(
                    target=self._run_u2net_parallel,
                    args=(imgs_bgr, masks_holder, u2net_done),
                    daemon=True,
                    name=f"U2NetThread-{trigger_id}",
                )
                u2net_thread.start()

                # ── PatchCore inference ────────────────────────────────
                patchcore_results = run_inspection_batch(
                    batch_images,
                    self.model_engine,
                    self.memory_bank,
                    self.threshold,
                )

                # ── Đợi U2Net ─────────────────────────────────────────
                u2net_done.wait(timeout=2.0)
                if not u2net_done.is_set():
                    print(f"[{self.name}] ⚠️  U2Net timeout trigger={trigger_id}")

                masks = masks_holder[0] or [None] * len(batch_images)

                # ── Merge: score/OK-NG dựa trên patch trong mask ──────
                results = merge_patchcore_with_masks(
                    patchcore_results, masks, self.threshold
                )

                for cam_id, result in enumerate(results):
                    if result is not None:
                        result["trigger_id"] = trigger_id
                        result["cam_id"]     = cam_id

                t_total = (time.perf_counter() - t_start) * 1000
                print(f"[{self.name}] trigger={trigger_id} tổng: {t_total:.1f}ms")

                # ── Log kết quả cuối ───────────────────────────────────
                print(f"\n--- KẾT QUẢ CUỐI (trigger={trigger_id}) ---")
                for cam_id, r in enumerate(results):
                    if r is None:
                        print(f"  cam[{cam_id}] ⚫ SKIP")
                    else:
                        has_mask = r["object_mask"] is not None
                        print(f"  cam[{cam_id}] {r['result']}  "
                              f"score_in_mask={r['score']:.4f}  "
                              f"mask={'yes' if has_mask else 'no'}  "
                              f"boxes={len(r['anomaly_info']['boxes'])}")

                self.result_output_queue.put((trigger_id, results))

            except Exception as e:
                print(f"[{self.name}] ⚠️  Lỗi trigger {trigger_id}: {e}")
                self.result_output_queue.put((trigger_id, {"error": str(e)}))

            finally:
                self.frame_input_queue.task_done()

        print(f"[{self.name}] ✅ Kết thúc.")