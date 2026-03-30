import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torch2trt import torch2trt, TRTModule

# ─────────────────────────────────────────────
# 1. CẤU HÌNH HỆ THỐNG (ORIN FLEXIBLE OPTIMIZED)
# ─────────────────────────────────────────────
TRAIN_GOOD_FOLDER = Path("img")
# TEST_IMG_PATH     = Path("project/cam1_0002.jpg")
TEST_IMG_PATH     = Path("11.jpg")

# "Standard Workspace": Mọi ảnh (2448x600, 2048x500...) đều đưa về chuẩn này để so sánh
STD_W, STD_H      = 224, 224 
TARGET_BANK_SIZE  = 8000 
TRT_ENGINE_PATH   = "backbone_flex_224x224.pth"
BANK_CACHE_PATH   = "memory_bank.pt" 

device = torch.device("cuda")
dtype  = torch.float16

# Hằng số chuẩn hóa GPU
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

# ─────────────────────────────────────────────
# 2. ĐỊNH NGHĨA MODEL (HỖ TRỢ DYNAMIC SHAPE)
# ─────────────────────────────────────────────
class FlexibleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.DEFAULT
        model = wide_resnet50_2(weights=weights).eval()
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)   
        f3 = self.layer3(f2)  

        # Local Smoothing
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        
        # Nội suy f3 theo f2 (Thích ứng với mọi tỉ lệ ảnh)
        f3_up = F.interpolate(f3, size=(f2.shape[2], f2.shape[3]), mode="bilinear", align_corners=False)
        fmap  = torch.cat([f2, f3_up], dim=1) 
        
        # Output: [Patches, 1536]
        return fmap.reshape(fmap.shape[0], 1536, -1).permute(0, 2, 1).reshape(-1, 1536)

# ─────────────────────────────────────────────
# 3. QUẢN LÝ ENGINE (BUILD/LOAD)
# ─────────────────────────────────────────────
def get_engine():
    if os.path.exists(TRT_ENGINE_PATH):
        print(f"🚀 Loading Flexible Engine ({STD_W}x{STD_H})...")
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(TRT_ENGINE_PATH))
        return model_trt
    else:
        print(f"⏳ Đang khởi tạo Engine mới cho Orin (mất ~3 phút)...")
        model_raw = FlexibleBackbone().to(device).half().eval()
        dummy = torch.randn((1, 3, STD_H, STD_W), device=device, dtype=dtype)
        # Tối ưu hóa bộ nhớ 1GB cho TensorRT
        model_trt = torch2trt(model_raw, [dummy], fp16_mode=True, max_workspace_size=1 << 30)
        torch.save(model_trt.state_dict(), TRT_ENGINE_PATH)
        return model_trt

model_engine = get_engine()

# ─────────────────────────────────────────────
# 4. TIỀN XỬ LÝ LINH HOẠT
# ─────────────────────────────────────────────
def preprocess_any_size(path):
    # Load ảnh bất kỳ (2448x600, 2048x500...)
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    
    # Đưa về chuẩn Standard Workspace để đồng bộ với Bank
    img_res = img.resize((STD_W, STD_H), Image.LANCZOS)
    
    img_t = torch.from_numpy(np.array(img_res)).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    return (img_t / 255.0 - MEAN) / STD, (orig_w, orig_h)

# ─────────────────────────────────────────────
# 5. XÂY DỰNG DYNAMIC BANK (CORESET)
# ─────────────────────────────────────────────
def build_bank():
    print(f"── Đang xây dựng Bank từ tập huấn luyện (Good images) ──")
    all_feats = []
    image_files = [f for f in TRAIN_GOOD_FOLDER.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    with torch.inference_mode():
        for pth in sorted(image_files):
            img_t, _ = preprocess_any_size(pth)
            feat = model_engine(img_t)
            all_feats.append(feat.cpu())

    full_bank = torch.cat(all_feats, dim=0)
    
    # Dynamic Coreset: Lấy mẫu hệ thống để bao phủ toàn bộ đặc trưng
    if full_bank.shape[0] > TARGET_BANK_SIZE:
        indices = torch.linspace(0, full_bank.shape[0] - 1, steps=TARGET_BANK_SIZE).long()
        bank = full_bank[indices]
    else:
        bank = full_bank
        

    print(f"💾 Đang lưu Memory Bank vào: {BANK_CACHE_PATH}")
    torch.save(bank, BANK_CACHE_PATH)
        
    return bank.to(device).contiguous()





# Khởi tạo Bank
memory_bank = build_bank()
print(f"✅ Dynamic Bank Ready: {memory_bank.shape}")

# ─────────────────────────────────────────────
# 6. VÒNG LẶP KIỂM TRA (INFERENCE)
# ─────────────────────────────────────────────
def run_inspection(img_path, threshold=14.0):
    # 1. Tiền xử lý (Thích ứng mọi kích thước)
    img_t, (w, h) = preprocess_any_size(img_path)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        # 2. Trích xuất đặc trưng qua TensorRT
        features = model_engine(img_t) # [3136, 1536]

        # 3. Tính khoảng cách cực nhanh (Matrix Mult)
        a_sq = (features**2).sum(dim=1, keepdim=True)
        b_sq = (memory_bank**2).sum(dim=1).unsqueeze(0)
        dist_sq = a_sq + b_sq - 2.0 * torch.mm(features, memory_bank.t())
        
        # 4. Tìm điểm lỗi cao nhất
        dist_min, _ = torch.min(dist_sq.clamp(min=1e-6), dim=1)
        s_star = torch.max(dist_min.sqrt()).item()

    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    
    result = "NG (LỖI)" if s_star > threshold else "OK (ĐẠT)"
    
    print(f"\n--- KẾT QUẢ KIỂM TRA ---")
    print(f"🖼️ Ảnh gốc: {w}x{h}")
    print(f"⏱️ Tốc độ: {ms:.2f} ms")
    print(f"📊 Anomaly Score: {s_star:.4f}")
    print(f"📢 Kết luận: {result}")
    return s_star


def auto_threshold(bank, safety_factor=1.1):
    print("📏 Đang tự động tính toán ngưỡng tối ưu...")
    scores = []
    image_files = list(TRAIN_GOOD_FOLDER.glob('*'))[:20] # Lấy thử 20 ảnh mẫu
    for pth in image_files:
        s = run_inspection(pth, threshold=999) # Chạy không lấy kết luận
        scores.append(s)
    
    suggested = max(scores) * safety_factor
    print(f"🎯 Ngưỡng gợi ý (Threshold): {suggested:.2f}")
    return suggested



# ─────────────────────────────────────────────
# 7. CHẠY THỬ NGHIỆM
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Ép xung GPU Jetson trước khi đo tốc độ
    # Lưu ý: Chạy lệnh `sudo jetson_clocks` ngoài Terminal
    
    # Warm-up
    print("🔥 Warming up...")
    dummy_path = next(TRAIN_GOOD_FOLDER.iterdir())
    for _ in range(3): _ = run_inspection(dummy_path)
    
    # Test file thực tế
    if TEST_IMG_PATH.exists():
        run_inspection(TEST_IMG_PATH)
    else:
        print(f"⚠️ Không tìm thấy file test tại {TEST_IMG_PATH}")