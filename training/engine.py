import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torch2trt import torch2trt, TRTModule




# "Standard Workspace": Mọi ảnh (2448x600, 2048x500...) đều đưa về chuẩn này để so sánh
STD_W, STD_H      = 224, 224 
TARGET_BANK_SIZE  = 8000 
TRT_ENGINE_PATH   = "/home/m2m/Documents/Bottle-inspection/Application/backbone_flex_224x224.pth"

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
# QUẢN LÝ ENGINE (BUILD/LOAD)
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

