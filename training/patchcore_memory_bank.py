import torch
import numpy as np
from pathlib import Path
from PIL import Image
from training.engine import get_engine
from core.path_manager import BASE_DIR


BANK_CACHE_PATH   = "memory_bank.pt" 
TRAIN_GOOD_FOLDER = Path("/home/m2m/Documents/Bottle-inspection/Application/project")
# TEST_IMG_PATH     = Path("data_cap/fails/0.png")

device = torch.device("cuda")
dtype  = torch.float16

# Hằng số chuẩn hóa GPU
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)


# "Standard Workspace": Mọi ảnh (2448x600, 2048x500...) đều đưa về chuẩn này để so sánh
STD_W, STD_H      =  224, 224
TARGET_BANK_SIZE  = 8000 




# ─────────────────────────────────────────────
# TIỀN XỬ LÝ LINH HOẠT
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
# XÂY DỰNG DYNAMIC BANK (CORESET)
# ─────────────────────────────────────────────
def build_bank():
    model_engine = get_engine(STD_W, STD_H)

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

# # Khởi tạo Bank
# memory_bank = build_bank()
# print(f"✅ Dynamic Bank Ready: {memory_bank.shape}")