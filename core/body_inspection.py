<<<<<<< HEAD
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




=======
>>>>>>> 5fe2763 (update 2703)
import threading
import queue
import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
<<<<<<< HEAD
=======
import cv2
>>>>>>> 5fe2763 (update 2703)

try:
    from torch2trt import torch2trt, TRTModule
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

BANK_CACHE_PATH  = "memory_bank.pt"
TRT_ENGINE_PATH  = "backbone_flex_224x224.pth"
STD_W, STD_H     = 224, 224

<<<<<<< HEAD
=======
# WideResNet50 backbone → feature map 28x28 = 784 patches
FEAT_H = 28
FEAT_W = 28

>>>>>>> 5fe2763 (update 2703)
device   = torch.device("cuda")
dtype    = torch.float16

MEAN     = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD_NORM = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)


# =========================================================================== #
#  TIỀN XỬ LÝ (CPU)                                                           #
# =========================================================================== #
def _to_pil_rgb(img_input) -> tuple:
<<<<<<< HEAD
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
=======
    """
    Chuyển về PIL RGB + lấy kích thước gốc. Chạy trên CPU thread.
    Hỗ trợ:
      - Path / str          : đọc file từ disk
      - numpy RGB  (H,W,3)  : camera trả về RGB trực tiếp → dùng luôn
      - numpy RGBA (H,W,4)  : bỏ alpha channel
      - numpy Gray (H,W)    : grayscale → convert RGB
      - PIL Image           : giữ nguyên
    LƯU Ý: Camera này trả về RGB (không phải BGR) – KHÔNG flip channel.
    """
    if isinstance(img_input, (str, Path)):
        img = Image.open(img_input).convert("RGB")

    elif isinstance(img_input, np.ndarray):
        arr = np.ascontiguousarray(img_input)
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            c = arr.shape[2]
            if c == 3:
                # Camera trả về RGB trực tiếp – KHÔNG flip
                img = Image.fromarray(arr, mode="RGB")
            elif c == 4:
                # RGBA – bỏ alpha
                img = Image.fromarray(arr, mode="RGBA").convert("RGB")
            else:
                img = Image.fromarray(arr).convert("RGB")
        else:
            img = Image.fromarray(arr).convert("RGB")
    else:
        img = img_input.convert("RGB")

    orig_size = img.size                                   # (w, h)
    img_res   = img.resize((STD_W, STD_H), Image.LANCZOS)
    arr_out   = np.array(img_res, dtype=np.uint8)         # H W C, uint8 RGB
    return arr_out, orig_size


def _arr_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 HWC numpy (RGB) → normalized float16 [1,C,H,W] GPU tensor."""
>>>>>>> 5fe2763 (update 2703)
    t = (
        torch.from_numpy(arr)
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=dtype)
    )
    return (t / 255.0 - MEAN) / STD_NORM


# =========================================================================== #
<<<<<<< HEAD
=======
#  ANOMALY MAP – heatmap + bounding boxes vùng bất thường                     #
# =========================================================================== #
def _build_anomaly_map(patch_scores: torch.Tensor,
                       orig_w: int, orig_h: int,
                       threshold: float) -> dict:
    """
    Từ score từng patch → heatmap full-res + bounding boxes vùng bất thường.

    patch_scores : [784]  fp32 tensor – score từng patch
    orig_w, orig_h: kích thước ảnh gốc từ camera
    threshold    : ngưỡng anomaly

    Trả về dict:
        heatmap_gray  : numpy (orig_h, orig_w) uint8  – 0..255 để visualize
        anomaly_mask  : numpy (orig_h, orig_w) bool   – True = vùng lỗi
        boxes         : list[(x1,y1,x2,y2)]  tọa độ trên ảnh gốc
        patch_scores_2d: numpy (FEAT_H, FEAT_W) float32 – debug
    """
    # 1. Reshape về 2D feature map (28x28)
    scores_2d = patch_scores.cpu().float().numpy().reshape(FEAT_H, FEAT_W)

    # 2. Upsample về kích thước ảnh gốc
    heatmap_full = cv2.resize(
        scores_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )  # (orig_h, orig_w) float32

    # 3. Normalize về 0-255 để visualize
    s_min, s_max = heatmap_full.min(), heatmap_full.max()
    if s_max > s_min:
        heatmap_norm = ((heatmap_full - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        heatmap_norm = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # 4. Mask vùng bất thường (score > threshold)
    anomaly_mask = (heatmap_full > threshold).astype(np.uint8)

    # 5. Tìm bounding boxes từ contours
    boxes = []
    if anomaly_mask.any():
        contours, _ = cv2.findContours(
            anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:   # bỏ vùng quá nhỏ (< 100 px²)
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((x, y, x + bw, y + bh))

    return {
        "heatmap_gray"   : heatmap_norm,                  # (H, W) uint8
        "anomaly_mask"   : anomaly_mask.astype(bool),     # (H, W) bool
        "boxes"          : boxes,                         # [(x1,y1,x2,y2), ...]
        "patch_scores_2d": scores_2d,                     # (28,28) float32
    }


def draw_anomaly_overlay(frame_bgr: np.ndarray, anomaly_info: dict,
                         alpha: float = 0.5) -> np.ndarray:
    """
    Vẽ heatmap + bounding boxes lên frame BGR để hiển thị.

    frame_bgr   : numpy (H, W, 3) BGR
    anomaly_info: dict từ _build_anomaly_map
    alpha       : độ trong suốt heatmap (0=ẩn, 1=đặc)
    Trả về frame BGR mới đã overlay.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # 1. Heatmap màu: xanh=bình thường, đỏ=lỗi
    heatmap_color = cv2.applyColorMap(anomaly_info["heatmap_gray"], cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))

    # 2. Blend heatmap lên ảnh gốc
    cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0, img)

    # 3. Vẽ bounding boxes vùng lỗi
    for (x1, y1, x2, y2) in anomaly_info["boxes"]:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "DEFECT", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    return img


# =========================================================================== #
>>>>>>> 5fe2763 (update 2703)
#  INFERENCE – TỪNG ẢNH MỘT (TRT batch=1), PREPROCESS SONG SONG              #
# =========================================================================== #
def run_inspection_batch(images: list, model_engine, memory_bank,
                         threshold: float = 14.0) -> list:
    """
    - Preprocess song song trên CPU (ThreadPoolExecutor)
    - Forward từng ảnh một qua TRT engine (batch=1)
<<<<<<< HEAD
    - Tính distance 1 lần cho tất cả features

    images  : list[PIL.Image | numpy.ndarray | None]  – len = số cam
    Trả về  : list[dict | None]  – cùng thứ tự cam_id
    """
=======
    - Tính distance 1 lần cho tất cả features (fp32 tránh overflow)
    - Lấy MAX score trên tất cả patches của mỗi ảnh
    - Tính heatmap + bounding boxes vùng bất thường

    images  : list[PIL.Image | numpy.ndarray | None]  – len = số cam
    Trả về  : list[dict | None]  – cùng thứ tự cam_id

    Mỗi dict kết quả:
        score        : float  – anomaly score tổng
        is_ok        : bool
        result       : str    – "OK (ĐẠT)" | "NG (LỖI)"
        time_ms      : float
        img_size     : (w, h)
        anomaly_info : {
            heatmap_gray   : numpy (H,W) uint8
            anomaly_mask   : numpy (H,W) bool
            boxes          : [(x1,y1,x2,y2), ...]  ← tọa độ vùng lỗi ảnh gốc
            patch_scores_2d: numpy (28,28) float32
        }
    """

    print('[BodyInspection] run inspection')
>>>>>>> 5fe2763 (update 2703)
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
<<<<<<< HEAD
    # preprocessed: [(cam_id, arr, orig_size), ...]  – thứ tự không đảm bảo
    preprocessed.sort(key=lambda x: x[0])            # sort lại theo cam_id

    # ── Forward từng ảnh qua TRT (batch=1) ────────────────────────────────
    features_list = []   # [(cam_id, orig_size, feat)]
=======

    preprocessed.sort(key=lambda x: x[0])

    # ── Forward từng ảnh qua TRT (batch=1) ────────────────────────────────
    features_list = []  # [(cam_id, orig_size, feat_tensor)]
>>>>>>> 5fe2763 (update 2703)

    torch.cuda.synchronize()
    with torch.inference_mode():
        for cam_id, arr, orig_size in preprocessed:
<<<<<<< HEAD
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
=======
            img_t = _arr_to_tensor(arr)          # [1, C, H, W]
            feat  = model_engine(img_t)          # [784, 1536]
            features_list.append((cam_id, orig_size, feat))

    # ── Tính distance (fp32 tránh overflow) ───────────────────────────────
    with torch.inference_mode():
        features_all    = torch.cat([f for _, _, f in features_list], dim=0)  # [K*784, 1536]
        patches_per_img = features_list[0][2].shape[0]   # 784
        K               = len(features_list)

        feat_f32 = features_all.float()
        bank_f32 = memory_bank.float()

        a_sq    = (feat_f32 ** 2).sum(dim=1, keepdim=True)
        b_sq    = (bank_f32 ** 2).sum(dim=1).unsqueeze(0)
        dist_sq = (
            a_sq + b_sq - 2.0 * torch.mm(feat_f32, bank_f32.t())
        ).clamp(min=1e-6)

        # Score từng patch: [K*784]
        patch_dist = dist_sq.min(dim=1).values.sqrt()

        # Reshape: [K, 784]
        patch_dist = patch_dist.reshape(K, patches_per_img)

        # Anomaly score mỗi ảnh = MAX patch
        scores = patch_dist.max(dim=1).values                     # [K]

    torch.cuda.synchronize()
    ms_total = (time.perf_counter() - t0) * 1000
    ms_each  = ms_total / K

    # ── Gắn kết quả + anomaly map ─────────────────────────────────────────
    for k, (cam_id, (w, h), _) in enumerate(features_list):
        s     = scores[k].item()
        label = "NG (LỖI)" if s > threshold else "OK (ĐẠT)"

        anomaly_info = _build_anomaly_map(
            patch_scores = patch_dist[k],   # [784]
            orig_w       = w,
            orig_h       = h,
            threshold    = threshold,
        )

        results[cam_id] = {
            "score"       : s,
            "is_ok"       : s <= threshold,
            "result"      : label,
            "time_ms"     : ms_each,
            "img_size"    : (w, h),
            "anomaly_info": anomaly_info,
>>>>>>> 5fe2763 (update 2703)
        }

    # ── Log ───────────────────────────────────────────────────────────────
    print(f"\n--- KẾT QUẢ ({K}/{len(images)} ảnh | tổng {ms_total:.1f}ms | mỗi ảnh ~{ms_each:.1f}ms) ---")
    for cam_id, r in enumerate(results):
        if r is None:
            print(f"  cam[{cam_id}] ⚫ SKIP (None)")
        else:
<<<<<<< HEAD
            w, h = r["img_size"]
            print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}  {w}x{h}")
=======
            w, h  = r["img_size"]
            boxes = r["anomaly_info"]["boxes"]
            print(f"  cam[{cam_id}] {r['result']}  score={r['score']:.4f}  "
                  f"{w}x{h}  defect_boxes={len(boxes)}")
>>>>>>> 5fe2763 (update 2703)

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
<<<<<<< HEAD
    Mỗi img: PIL.Image | numpy.ndarray | None.
    Gửi (trigger_id, list[dict|None]) ra result_output_queue – thứ tự cam_id.
=======
    Mỗi img: PIL.Image | numpy.ndarray (RGB hoặc RGBA) | None.
    Gửi (trigger_id, list[dict|None]) ra result_output_queue – thứ tự cam_id.

    Mỗi dict kết quả có thêm "anomaly_info" chứa:
        boxes          : [(x1,y1,x2,y2), ...]  – tọa độ vùng lỗi trên ảnh gốc
        heatmap_gray   : numpy (H,W) uint8      – để visualize
        anomaly_mask   : numpy (H,W) bool
        patch_scores_2d: numpy (28,28) float32

    Dùng draw_anomaly_overlay(frame_bgr, result["anomaly_info"]) để vẽ lên ảnh.
>>>>>>> 5fe2763 (update 2703)
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

<<<<<<< HEAD
=======
        self._warmup()

    def _warmup(self, n: int = 3):
        print(f"[{self.name}] 🔥 Warming up TRT engine ({n} lần)...")
        dummy = torch.zeros((1, 3, STD_H, STD_W), device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(n):
                _ = self.model_engine(dummy)
        torch.cuda.synchronize()
        print(f"[{self.name}] ✅ Warm-up xong.")

>>>>>>> 5fe2763 (update 2703)
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