# # lấy ảnh từ camera, xử lý ảnh
# import threading
# import os
# import time
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
# from torch2trt import torch2trt, TRTModule


# BANK_CACHE_PATH   = "memory_bank.pt" 

# # "Standard Workspace": Mọi ảnh (2448x600, 2048x500...) đều đưa về chuẩn này để so sánh
# STD_W, STD_H      = 224, 224 
# TARGET_BANK_SIZE  = 8000 
# TRT_ENGINE_PATH   = "backbone_flex_224x224.pth"

# device = torch.device("cuda")
# dtype  = torch.float16

# # Hằng số chuẩn hóa GPU
# MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
# STD  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)







        



# def preprocess_any_size(path):
#     # Load ảnh bất kỳ (2448x600, 2048x500...)
#     img = Image.open(path).convert("RGB")
#     orig_w, orig_h = img.size
    
#     # Đưa về chuẩn Standard Workspace để đồng bộ với Bank
#     img_res = img.resize((STD_W, STD_H), Image.LANCZOS)
    
#     img_t = torch.from_numpy(np.array(img_res)).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
#     return (img_t / 255.0 - MEAN) / STD, (orig_w, orig_h)



# def run_inspection(img_path, threshold=14.0):
#     # 1. Tiền xử lý (Thích ứng mọi kích thước)
#     img_t, (w, h) = preprocess_any_size(img_path)

#     torch.cuda.synchronize()
#     t0 = time.perf_counter()

#     with torch.inference_mode():
#         # 2. Trích xuất đặc trưng qua TensorRT
#         features = self.model_engine(img_t) 

#         # 3. Tính khoảng cách cực nhanh (Matrix Mult)
#         a_sq = (features**2).sum(dim=1, keepdim=True)
#         b_sq = (self.memory_bank**2).sum(dim=1).unsqueeze(0)
#         dist_sq = a_sq + b_sq - 2.0 * torch.mm(features, self.memory_bank.t())
        
#         # 4. Tìm điểm lỗi cao nhất
#         dist_min, _ = torch.min(dist_sq.clamp(min=1e-6), dim=1)
#         s_star = torch.max(dist_min.sqrt()).item()

#     torch.cuda.synchronize()
#     ms = (time.perf_counter() - t0) * 1000
    
#     result = "NG (LỖI)" if s_star > threshold else "OK (ĐẠT)"
    
#     print(f"\n--- KẾT QUẢ KIỂM TRA ---")
#     print(f"🖼️ Ảnh gốc: {w}x{h}")
#     print(f"⏱️ Tốc độ: {ms:.2f} ms")
#     print(f"📊 Anomaly Score: {s_star:.4f}")
#     print(f"📢 Kết luận: {result}")
#     return s_star

# def load_system():
#     # 1. Load/Build Engine
#     if os.path.exists(TRT_ENGINE_PATH):
#         print(f"🚀 Loading Engine: {TRT_ENGINE_PATH}")
#         model_engine = TRTModule()
#         model_engine.load_state_dict(torch.load(TRT_ENGINE_PATH))
#         return None, None
 
#     # 2. Load/Build Memory Bank
#     if os.path.exists(BANK_CACHE_PATH):
#         print(f"📂 Loading Memory Bank: {BANK_CACHE_PATH}")
#         bank = torch.load(BANK_CACHE_PATH).to(device).contiguous()
#         return None, None

#     return model_engine, bank


# class BodyWorker(threading.Thread):
#     def __init__(self, frame_input_queue, stop_event):
#         super().__init__()

#         self.frame_input_queue = frame_input_queue
#         self.stop_event = stop_event

#         self.model_engine, self.memory_bank = load_system()

# -------------------------------------------------------------------
# ------------------------------------------------------------------

# # truyen batch size la 4 anh nhung do model build voi batch size la 1
# import threading
# import queue
# import os
# import time
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# try:
#     from torch2trt import torch2trt, TRTModule
#     HAS_TRT = True
# except ImportError:
#     HAS_TRT = False

# BANK_CACHE_PATH  = "memory_bank.pt"
# TRT_ENGINE_PATH  = "backbone_flex_224x224.pth"
# STD_W, STD_H     = 224, 224
# TARGET_BANK_SIZE = 8000

# device   = torch.device("cuda")
# dtype    = torch.float16

# MEAN     = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
# STD_NORM = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)


# # =========================================================================== #
# #  TIỀN XỬ LÝ                                                                 #
# # =========================================================================== #
# def preprocess_any_size(path_or_pil):
#     """Trả về tensor [1, C, H, W] đã chuẩn hóa + kích thước gốc.
#     Hỗ trợ: đường dẫn file | PIL.Image | numpy.ndarray (BGR hoặc RGB)
#     """
#     if isinstance(path_or_pil, (str, Path)):
#         img = Image.open(path_or_pil).convert("RGB")
#     elif isinstance(path_or_pil, np.ndarray):
#         # Camera (OpenCV) trả về BGR uint8 → đổi sang RGB
#         if path_or_pil.ndim == 3 and path_or_pil.shape[2] == 3:
#             img = Image.fromarray(path_or_pil[..., ::-1].copy())  # BGR → RGB
#         else:
#             img = Image.fromarray(path_or_pil)                    # grayscale / RGB sẵn
#     else:
#         img = path_or_pil.convert("RGB")
#     orig_w, orig_h = img.size
#     img_res = img.resize((STD_W, STD_H), Image.LANCZOS)
#     img_t = (
#         torch.from_numpy(np.array(img_res))
#         .to(device)
#         .permute(2, 0, 1)
#         .unsqueeze(0)
#         .to(dtype=dtype)
#     )
#     return (img_t / 255.0 - MEAN) / STD_NORM, (orig_w, orig_h)


# def preprocess_batch(images: list):
#     """
#     Nhận list ảnh (PIL | đường dẫn | None), chỉ xử lý ảnh hợp lệ.
#     Trả về:
#         batch_t    : Tensor [K, C, H, W]  – K = số ảnh không None
#         orig_sizes : list[(w, h)]          – độ dài K
#         valid_indices: list[int]           – vị trí cam_id có ảnh thật
#     """
#     tensors       = []
#     orig_sizes    = []
#     valid_indices = []

#     for cam_id, img in enumerate(images):
#         if img is None:
#             continue                          # bỏ qua cam không có ảnh
#         t, size = preprocess_any_size(img)
#         tensors.append(t)
#         orig_sizes.append(size)
#         valid_indices.append(cam_id)

#     if not tensors:
#         return None, [], []                   # toàn None → không có gì inference

#     batch_t = torch.cat(tensors, dim=0)       # [K, C, H, W]
#     return batch_t, orig_sizes, valid_indices


# # =========================================================================== #
# #  INFERENCE BATCH                                                             #
# # =========================================================================== #
# def run_inspection_batch(images: list, model_engine, memory_bank,
#                          threshold: float = 14.0) -> list:
#     """
#     Chạy anomaly detection cho cả batch trong 1 lần forward pass.
#     Ảnh None được bỏ qua – kết quả của chúng sẽ là None trong list trả về.

#     images  : list[PIL.Image | None]  – giữ nguyên vị trí cam_id
#     Trả về  : list[dict | None]       – cùng độ dài với images
#         dict  = { score, is_ok, result, time_ms, img_size }
#         None  = cam không có ảnh
#     """
#     batch_t, orig_sizes, valid_indices = preprocess_batch(images)

#     # Khởi tạo kết quả với None cho toàn bộ vị trí
#     results = [None] * len(images)

#     # Nếu không có ảnh nào hợp lệ thì trả về ngay
#     if batch_t is None:
#         print("⚠️  Batch toàn None – bỏ qua inference.")
#         return results

#     K = batch_t.shape[0]

#     torch.cuda.synchronize()
#     t0 = time.perf_counter()

#     with torch.inference_mode():
#         # 1 lần forward → features [K, D]
#         features = model_engine(batch_t)

#         # L2² distance: [K, M]
#         a_sq    = (features ** 2).sum(dim=1, keepdim=True)
#         b_sq    = (memory_bank ** 2).sum(dim=1).unsqueeze(0)
#         dist_sq = (a_sq + b_sq - 2.0 * torch.mm(features, memory_bank.t())).clamp(min=1e-6)

#         scores = dist_sq.min(dim=1).values.sqrt()   # [K]

#     torch.cuda.synchronize()
#     ms_total = (time.perf_counter() - t0) * 1000
#     ms_each  = ms_total / K

#     # Gắn kết quả vào đúng vị trí cam_id
#     for k, cam_id in enumerate(valid_indices):
#         s     = scores[k].item()
#         w, h  = orig_sizes[k]
#         label = "NG (LỖI)" if s > threshold else "OK (ĐẠT)"
#         results[cam_id] = {
#             "score":    s,
#             "is_ok":    s <= threshold,
#             "result":   label,
#             "time_ms":  ms_each,
#             "img_size": (w, h),
#         }

#     # Log tổng hợp
#     print(f"\n--- KẾT QUẢ BATCH ({K}/{len(images)} ảnh | tổng {ms_total:.1f}ms) ---")
#     for cam_id, r in enumerate(results):
#         if r is None:
#             print(f"  cam[{cam_id}] ⚫ SKIP (None)")
#         else:
#             w, h = r["img_size"]
#             print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}  {w}x{h}  {r['time_ms']:.1f}ms")

#     return results


# # =========================================================================== #
# #  LOAD HỆ THỐNG                                                              #
# # =========================================================================== #
# def load_system():
#     model_engine = None
#     bank         = None

#     if HAS_TRT and os.path.exists(TRT_ENGINE_PATH):
#         print(f"🚀 Loading TRT Engine: {TRT_ENGINE_PATH}")
#         model_engine = TRTModule()
#         model_engine.load_state_dict(torch.load(TRT_ENGINE_PATH, map_location=device))
#         model_engine.eval()
#     else:
#         print("⚠️  Không tìm thấy TRT engine.")

#     if os.path.exists(BANK_CACHE_PATH):
#         print(f"📂 Loading Memory Bank: {BANK_CACHE_PATH}")
#         bank = torch.load(BANK_CACHE_PATH, map_location=device).to(dtype).contiguous()
#         print(f"   Bank shape: {bank.shape}")
#     else:
#         print("⚠️  Memory Bank chưa có.")

#     return model_engine, bank


# # =========================================================================== #
# #  BODY WORKER                                                                 #
# # =========================================================================== #
# class BodyWorker(threading.Thread):
#     """
#     Nhận batch ảnh từ frame_input_queue, chạy anomaly detection 1 lần forward,
#     gửi list kết quả ra result_output_queue.

#     frame_input_queue  : Queue – item là tuple (trigger_id, [img1, img2, img3, img4])
#                          Đặt None để báo dừng.
#     result_output_queue: Queue – worker đặt tuple (trigger_id, list[dict]) vào đây.
#     stop_event         : threading.Event
#     threshold          : float
#     """

#     def __init__(
#         self,
#         frame_input_queue:   queue.Queue,
#         result_output_queue: queue.Queue,
#         stop_event:          threading.Event,
#         threshold:           float = 14.0,
#     ):
#         super().__init__(daemon=True, name="BodyWorker")
#         self.frame_input_queue   = frame_input_queue
#         self.result_output_queue = result_output_queue
#         self.stop_event          = stop_event
#         self.threshold           = threshold

#         self.model_engine, self.memory_bank = load_system()

#         if self.model_engine is None:
#             raise RuntimeError("❌ Không load được model engine.")
#         if self.memory_bank is None:
#             raise RuntimeError("❌ Không load được memory bank.")

#     # ----------------------------------------------------------------------- #
#     def run(self):
#         print(f"[{self.name}] ▶ Bắt đầu. Chờ batch từ queue …")

#         while not self.stop_event.is_set():
#             try:
#                 item = self.frame_input_queue.get(timeout=0.5)
#             except queue.Empty:
#                 continue

#             # Sentinel – dừng worker
#             if item is None:
#                 print(f"[{self.name}] 🛑 Nhận lệnh dừng.")
#                 self.frame_input_queue.task_done()
#                 break

#             trigger_id, batch_images = item
#             # batch_images = [img1, img2, img3, img4]
#             # mỗi phần tử là PIL.Image hoặc None

#             try:
#                 results = run_inspection_batch(
#                     batch_images,
#                     self.model_engine,
#                     self.memory_bank,
#                     self.threshold,
#                 )
#                 # results[cam_id] = dict nếu có ảnh, None nếu ảnh là None
#                 for cam_id, result in enumerate(results):
#                     if result is not None:
#                         result["trigger_id"] = trigger_id
#                         result["cam_id"]     = cam_id

#                 self.result_output_queue.put((trigger_id, results))

#             except Exception as e:
#                 print(f"[{self.name}] ⚠️  Lỗi trigger {trigger_id}: {e}")
#                 self.result_output_queue.put((trigger_id, {"error": str(e)}))

#             finally:
#                 self.frame_input_queue.task_done()

#         print(f"[{self.name}] ✅ Kết thúc.")





# # -------------------------------------------------------------------
# #  [preprocess img1] → [forward] → [preprocess img2] → [forward] = ~141ms
# #   ↑ preprocess CPU block GPU giữa 2 lần forward

# import threading
# import queue
# import os
# import time
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image

# try:
#     from torch2trt import torch2trt, TRTModule
#     HAS_TRT = True
# except ImportError:
#     HAS_TRT = False

# BANK_CACHE_PATH  = "memory_bank.pt"
# TRT_ENGINE_PATH  = "backbone_flex_224x224.pth"
# STD_W, STD_H     = 224, 224

# device   = torch.device("cuda")
# dtype    = torch.float16

# MEAN     = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
# STD_NORM = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)


# # =========================================================================== #
# #  TIỀN XỬ LÝ                                                                 #
# # =========================================================================== #
# def preprocess_any_size(img_input):
#     """
#     Nhận: đường dẫn file | PIL.Image | numpy.ndarray (BGR từ OpenCV)
#     Trả về: tensor [1, C, H, W] đã chuẩn hóa, (orig_w, orig_h)
#     """
#     if isinstance(img_input, (str, Path)):
#         img = Image.open(img_input).convert("RGB")
#     elif isinstance(img_input, np.ndarray):
#         # OpenCV: BGR uint8 → PIL RGB
#         if img_input.ndim == 3 and img_input.shape[2] == 3:
#             img = Image.fromarray(img_input[..., ::-1].copy())  # BGR → RGB
#         else:
#             img = Image.fromarray(img_input)
#     else:
#         img = img_input.convert("RGB")

#     orig_w, orig_h = img.size
#     img_res = img.resize((STD_W, STD_H), Image.LANCZOS)
#     img_t = (
#         torch.from_numpy(np.array(img_res))
#         .to(device)
#         .permute(2, 0, 1)   # H W C → C H W
#         .unsqueeze(0)        # → [1, C, H, W]
#         .to(dtype=dtype)
#     )
#     return (img_t / 255.0 - MEAN) / STD_NORM, (orig_w, orig_h)


# # =========================================================================== #
# #  INFERENCE – TỪNG ẢNH MỘT (TRT batch=1)                                    #
# # =========================================================================== #
# def run_inspection_batch(images: list, model_engine, memory_bank,
#                          threshold: float = 14.0) -> list:
#     """
#     Forward từng ảnh một qua TRT engine (batch=1).
#     Ảnh None → bỏ qua, vị trí đó trong kết quả là None.

#     images  : list[PIL.Image | numpy.ndarray | None]  – len = số cam
#     Trả về  : list[dict | None]  – cùng thứ tự cam_id
#         dict = { score, is_ok, result, time_ms, img_size, trigger_id, cam_id }
#         None = cam không có ảnh
#     """
#     results = [None] * len(images)

#     torch.cuda.synchronize()
#     t0 = time.perf_counter()

#     features_list = []   # [(cam_id, (w,h), feat_tensor)]

#     with torch.inference_mode():
#         for cam_id, img in enumerate(images):
#             if img is None:
#                 continue
#             img_t, size = preprocess_any_size(img)   # [1, C, H, W]
#             feat = model_engine(img_t)                # [1, D]
#             features_list.append((cam_id, size, feat))

#     if not features_list:
#         print("⚠️  Batch toàn None – bỏ qua inference.")
#         return results

#     K = len(features_list)

#     # Tính distance 1 lần cho tất cả features đã có
#     with torch.inference_mode():
#         features = torch.cat([f for _, _, f in features_list], dim=0)  # [K, D]
#         a_sq    = (features ** 2).sum(dim=1, keepdim=True)             # [K, 1]
#         b_sq    = (memory_bank ** 2).sum(dim=1).unsqueeze(0)           # [1, M]
#         dist_sq = (a_sq + b_sq - 2.0 * torch.mm(features, memory_bank.t())).clamp(min=1e-6)
#         scores  = dist_sq.min(dim=1).values.sqrt()                     # [K]

#     torch.cuda.synchronize()
#     ms_total = (time.perf_counter() - t0) * 1000
#     ms_each  = ms_total / K

#     # Gắn kết quả về đúng vị trí cam_id
#     for k, (cam_id, (w, h), _) in enumerate(features_list):
#         s     = scores[k].item()
#         label = "NG (LỖI)" if s > threshold else "OK (ĐẠT)"
#         results[cam_id] = {
#             "score":    s,
#             "is_ok":    s <= threshold,
#             "result":   label,
#             "time_ms":  ms_each,
#             "img_size": (w, h),
#         }

#     # Log
#     print(f"\n--- KẾT QUẢ ({K}/{len(images)} ảnh | tổng {ms_total:.1f}ms) ---")
#     for cam_id, r in enumerate(results):
#         if r is None:
#             print(f"  cam[{cam_id}] ⚫ SKIP (None)")
#         else:
#             w, h = r["img_size"]
#             print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}  {w}x{h}  {r['time_ms']:.1f}ms")

#     return results


# # =========================================================================== #
# #  LOAD HỆ THỐNG                                                              #
# # =========================================================================== #
# def load_system():
#     model_engine = None
#     bank         = None

#     if HAS_TRT and os.path.exists(TRT_ENGINE_PATH):
#         print(f"🚀 Loading TRT Engine: {TRT_ENGINE_PATH}")
#         model_engine = TRTModule()
#         model_engine.load_state_dict(torch.load(TRT_ENGINE_PATH, map_location=device))
#         model_engine.eval()
#     else:
#         print("⚠️  Không tìm thấy TRT engine.")

#     if os.path.exists(BANK_CACHE_PATH):
#         print(f"📂 Loading Memory Bank: {BANK_CACHE_PATH}")
#         bank = torch.load(BANK_CACHE_PATH, map_location=device).to(dtype).contiguous()
#         print(f"   Bank shape: {bank.shape}")
#     else:
#         print("⚠️  Memory Bank chưa có.")

#     return model_engine, bank


# # =========================================================================== #
# #  BODY WORKER                                                                 #
# # =========================================================================== #
# class BodyWorker(threading.Thread):
#     """
#     Nhận (trigger_id, [img1, img2, img3, img4]) từ frame_input_queue.
#     Mỗi img có thể là PIL.Image | numpy.ndarray | None.
#     Forward từng ảnh qua TRT engine (batch=1), tính distance cùng lúc.
#     Gửi (trigger_id, list[dict|None]) ra result_output_queue – thứ tự theo cam_id.
#     """

#     def __init__(
#         self,
#         frame_input_queue:   queue.Queue,
#         result_output_queue: queue.Queue,
#         stop_event:          threading.Event,
#         threshold:           float = 14.0,
#     ):
#         super().__init__(daemon=True, name="BodyWorker")
#         self.frame_input_queue   = frame_input_queue
#         self.result_output_queue = result_output_queue
#         self.stop_event          = stop_event
#         self.threshold           = threshold

#         self.model_engine, self.memory_bank = load_system()

#         if self.model_engine is None:
#             raise RuntimeError("❌ Không load được model engine.")
#         if self.memory_bank is None:
#             raise RuntimeError("❌ Không load được memory bank.")

#     # ----------------------------------------------------------------------- #
#     def run(self):
#         print(f"[{self.name}] ▶ Bắt đầu. Chờ từ queue …")

#         while not self.stop_event.is_set():
#             try:
#                 item = self.frame_input_queue.get(timeout=0.5)
#             except queue.Empty:
#                 continue

#             if item is None:   # sentinel
#                 print(f"[{self.name}] 🛑 Nhận lệnh dừng.")
#                 self.frame_input_queue.task_done()
#                 break

#             trigger_id, batch_images = item
#             # batch_images = [img1, img2, img3, img4]  – mỗi phần tử có thể None

#             try:
#                 results = run_inspection_batch(
#                     batch_images,
#                     self.model_engine,
#                     self.memory_bank,
#                     self.threshold,
#                 )
#                 # Gắn meta – chỉ với cam có ảnh thật
#                 for cam_id, result in enumerate(results):
#                     if result is not None:
#                         result["trigger_id"] = trigger_id
#                         result["cam_id"]     = cam_id

#                 # results[cam_id] = dict | None  – đúng thứ tự cam
#                 self.result_output_queue.put((trigger_id, results))

#             except Exception as e:
#                 print(f"[{self.name}] ⚠️  Lỗi trigger {trigger_id}: {e}")
#                 self.result_output_queue.put((trigger_id, {"error": str(e)}))

#             finally:
#                 self.frame_input_queue.task_done()

#         print(f"[{self.name}] ✅ Kết thúc.")





# --------------------------------------------------------------------
# ThreadPoolExecutor (CPU):           GPU:
#   resize img1  ──┐
#   resize img2  ──┘ → arr1, arr2   →  forward(arr1) → feat1
#                                    →  forward(arr2) → feat2
#                                    →  distance(feat1, feat2, bank)




import threading
import queue
import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

try:
    from torch2trt import torch2trt, TRTModule
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

BANK_CACHE_PATH  = "memory_bank.pt"
TRT_ENGINE_PATH  = "backbone_flex_224x224.pth"
STD_W, STD_H     = 224, 224

device   = torch.device("cuda")
dtype    = torch.float16

MEAN     = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD_NORM = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)


# =========================================================================== #
#  TIỀN XỬ LÝ (CPU)                                                           #
# =========================================================================== #
def _to_pil_rgb(img_input) -> tuple:
    """Chuyển về PIL RGB + lấy kích thước gốc. Chạy trên CPU thread."""
    if isinstance(img_input, (str, Path)):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 3 and img_input.shape[2] == 3:
            img = Image.fromarray(img_input[..., ::-1].copy())  # BGR → RGB
        else:
            img = Image.fromarray(img_input)
    else:
        img = img_input.convert("RGB")
    orig_size = img.size                                  # (w, h)
    img_res   = img.resize((STD_W, STD_H), Image.LANCZOS)
    arr       = np.array(img_res, dtype=np.uint8)        # H W C, uint8
    return arr, orig_size


def _arr_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 HWC numpy → normalized float16 [1,C,H,W] GPU tensor."""
    t = (
        torch.from_numpy(arr)
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=dtype)
    )
    return (t / 255.0 - MEAN) / STD_NORM


# =========================================================================== #
#  INFERENCE – TỪNG ẢNH MỘT (TRT batch=1), PREPROCESS SONG SONG              #
# =========================================================================== #
def run_inspection_batch(images: list, model_engine, memory_bank,
                         threshold: float = 14.0) -> list:
    """
    - Preprocess song song trên CPU (ThreadPoolExecutor)
    - Forward từng ảnh một qua TRT engine (batch=1)
    - Tính distance 1 lần cho tất cả features

    images  : list[PIL.Image | numpy.ndarray | None]  – len = số cam
    Trả về  : list[dict | None]  – cùng thứ tự cam_id
    """
    results = [None] * len(images)

    # ── Lọc cam có ảnh thật ────────────────────────────────────────────────
    valid = [(cam_id, img) for cam_id, img in enumerate(images) if img is not None]
    if not valid:
        print("⚠️  Batch toàn None – bỏ qua inference.")
        return results

    # ── Preprocess song song trên CPU ──────────────────────────────────────
    t0 = time.perf_counter()

    def preprocess_one(args):
        cam_id, img = args
        arr, orig_size = _to_pil_rgb(img)
        return cam_id, arr, orig_size

    with ThreadPoolExecutor(max_workers=len(valid)) as pool:
        preprocessed = list(pool.map(preprocess_one, valid))
    # preprocessed: [(cam_id, arr, orig_size), ...]  – thứ tự không đảm bảo
    preprocessed.sort(key=lambda x: x[0])            # sort lại theo cam_id

    # ── Forward từng ảnh qua TRT (batch=1) ────────────────────────────────
    features_list = []   # [(cam_id, orig_size, feat)]

    torch.cuda.synchronize()
    with torch.inference_mode():
        for cam_id, arr, orig_size in preprocessed:
            img_t = _arr_to_tensor(arr)               # [1, C, H, W]
            feat  = model_engine(img_t)               # [1, D]
            features_list.append((cam_id, orig_size, feat))

    # ── Tính distance 1 lần ────────────────────────────────────────────────
    with torch.inference_mode():
        features = torch.cat([f for _, _, f in features_list], dim=0)  # [K, D]
        a_sq    = (features ** 2).sum(dim=1, keepdim=True)             # [K, 1]
        b_sq    = (memory_bank ** 2).sum(dim=1).unsqueeze(0)           # [1, M]
        dist_sq = (a_sq + b_sq - 2.0 * torch.mm(features, memory_bank.t())).clamp(min=1e-6)
        scores  = dist_sq.min(dim=1).values.sqrt()                     # [K]

    torch.cuda.synchronize()
    ms_total = (time.perf_counter() - t0) * 1000
    K        = len(features_list)
    ms_each  = ms_total / K

    # ── Gắn kết quả về đúng vị trí cam_id ────────────────────────────────
    for k, (cam_id, (w, h), _) in enumerate(features_list):
        s     = scores[k].item()
        label = "NG (LỖI)" if s > threshold else "OK (ĐẠT)"
        results[cam_id] = {
            "score":    s,
            "is_ok":    s <= threshold,
            "result":   label,
            "time_ms":  ms_each,
            "img_size": (w, h),
        }

    # ── Log ───────────────────────────────────────────────────────────────
    print(f"\n--- KẾT QUẢ ({K}/{len(images)} ảnh | tổng {ms_total:.1f}ms | mỗi ảnh ~{ms_each:.1f}ms) ---")
    for cam_id, r in enumerate(results):
        if r is None:
            print(f"  cam[{cam_id}] ⚫ SKIP (None)")
        else:
            w, h = r["img_size"]
            print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}  {w}x{h}")

    return results


# =========================================================================== #
#  LOAD HỆ THỐNG                                                              #
# =========================================================================== #
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
    Nhận (trigger_id, [img1, img2, img3, img4]) từ frame_input_queue.
    Mỗi img: PIL.Image | numpy.ndarray | None.
    Gửi (trigger_id, list[dict|None]) ra result_output_queue – thứ tự cam_id.
    """

    def __init__(
        self,
        frame_input_queue:   queue.Queue,
        result_output_queue: queue.Queue,
        stop_event:          threading.Event,
        threshold:           float = 14.0,
    ):
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

    def run(self):
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
                results = run_inspection_batch(
                    batch_images,
                    self.model_engine,
                    self.memory_bank,
                    self.threshold,
                )
                for cam_id, result in enumerate(results):
                    if result is not None:
                        result["trigger_id"] = trigger_id
                        result["cam_id"]     = cam_id

                self.result_output_queue.put((trigger_id, results))

            except Exception as e:
                print(f"[{self.name}] ⚠️  Lỗi trigger {trigger_id}: {e}")
                self.result_output_queue.put((trigger_id, {"error": str(e)}))

            finally:
                self.frame_input_queue.task_done()

        print(f"[{self.name}] ✅ Kết thúc.")