# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from training.engine import get_engine
# from core.path_manager import BASE_DIR


# MEMORY_BANK_FILENAME = "memory_bank.pt"

# device = torch.device("cuda")
# dtype  = torch.float16

# # Hằng số chuẩn hóa GPU
# MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
# STD  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

# # "Standard Workspace": Mọi ảnh đều đưa về chuẩn này để so sánh
# STD_W, STD_H     = 224, 224
# TARGET_BANK_SIZE = 8000


# # ─────────────────────────────────────────────
# # TIỀN XỬ LÝ LINH HOẠT
# # ─────────────────────────────────────────────
# def preprocess_any_size(path):
#     img = Image.open(path).convert("RGB")
#     orig_w, orig_h = img.size

#     img_res = img.resize((STD_W, STD_H), Image.LANCZOS)

#     img_t = (torch.from_numpy(np.array(img_res))
#              .to(device)
#              .permute(2, 0, 1)
#              .unsqueeze(0)
#              .to(dtype=dtype))
#     return (img_t / 255.0 - MEAN) / STD, (orig_w, orig_h)








# # ─────────────────────────────────────────────
# # Tính toán ngưỡng tối ưu
# # ─────────────────────────────────────────────
# def auto_threshold(bank, goods_dir: Path, safety_factor=1.1):
#     print("📏 Đang tự động tính toán ngưỡng tối ưu...")
#     scores = []
#     image_files = list(goods_dir.glob('*'))[:20] # Lấy thử 20 ảnh mẫu
#     for pth in image_files:
#         s = run_inspection(pth, threshold=999) # Chạy không lấy kết luận
#         scores.append(s)
    
#     suggested = max(scores) * safety_factor
#     print(f"🎯 Ngưỡng gợi ý (Threshold): {suggested:.2f}")
#     return suggested




# # ─────────────────────────────────────────────
# # XÂY DỰNG DYNAMIC BANK (CORESET)
# # ─────────────────────────────────────────────
# def build_bank(goods_dir: Path, output_dir: Path | None = None):
#     """
#     Xây dựng Memory Bank từ ảnh good và lưu vào output_dir.

#     Args:
#         goods_dir:  Thư mục chứa ảnh good.
#         output_dir: Thư mục lưu memory_bank.pt.
#                     Mặc định = goods_dir/../memory_bank/
#                     → projects/<name_project>/memory_bank/memory_bank.pt
#     """
#     goods_dir = Path(goods_dir)

#     # ── Xác định nơi lưu ─────────────────────────────────────────────────── #
#     if output_dir is None:
#         output_dir = goods_dir.parent / "memory_bank"
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     bank_path = output_dir / MEMORY_BANK_FILENAME

#     # ── Build ────────────────────────────────────────────────────────────── #
#     model_engine = get_engine(STD_W, STD_H)

#     print(f"── Đang xây dựng Bank từ tập huấn luyện (Good images) ──")
#     print(f"   📂 Nguồn  : {goods_dir}")
#     print(f"   💾 Lưu tại: {bank_path}")

#     all_feats = []
#     image_files = [
#         f for f in goods_dir.iterdir()
#         if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
#     ]

#     if not image_files:
#         raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {goods_dir}")

#     with torch.inference_mode():
#         for pth in sorted(image_files):
#             img_t, _ = preprocess_any_size(pth)
#             feat = model_engine(img_t)
#             all_feats.append(feat.cpu())

#     full_bank = torch.cat(all_feats, dim=0)

#     # Dynamic Coreset: lấy mẫu hệ thống để bao phủ toàn bộ đặc trưng
#     if full_bank.shape[0] > TARGET_BANK_SIZE:
#         indices = torch.linspace(
#             0, full_bank.shape[0] - 1, steps=TARGET_BANK_SIZE
#         ).long()
#         bank = full_bank[indices]
#     else:
#         bank = full_bank

#     torch.save(bank, bank_path)
#     print(f"✅ Memory Bank đã lưu: {bank_path}  (shape={tuple(bank.shape)})")

#     return bank.to(device).contiguous()


import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from training.engine import get_engine
from core.path_manager import BASE_DIR


MEMORY_BANK_FILENAME  = "memory_bank.pt"
PROJECT_INFO_FILENAME = "project_info.json"

device = torch.device("cuda")
dtype  = torch.float16

# Hằng số chuẩn hóa GPU
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

# "Standard Workspace": Mọi ảnh đều đưa về chuẩn này để so sánh
STD_W, STD_H     = 224, 224
TARGET_BANK_SIZE = 8000


# ─────────────────────────────────────────────
# TIỀN XỬ LÝ LINH HOẠT
# ─────────────────────────────────────────────
def preprocess_any_size(path):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size

    img_res = img.resize((STD_W, STD_H), Image.LANCZOS)

    img_t = (torch.from_numpy(np.array(img_res))
             .to(device)
             .permute(2, 0, 1)
             .unsqueeze(0)
             .to(dtype=dtype))
    return (img_t / 255.0 - MEAN) / STD, (orig_w, orig_h)


# ─────────────────────────────────────────────
# GHI threshold vào project_info.json
# ─────────────────────────────────────────────
def _save_threshold_to_json(project_root: Path, threshold: float):
    """Ghi threshold vào settings trong project_info.json."""
    json_path = project_root / PROJECT_INFO_FILENAME
    if not json_path.exists():
        print(f"⚠️  Không tìm thấy {json_path}, bỏ qua lưu threshold.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info.setdefault("settings", {})["threshold"] = round(float(threshold), 2)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print(f"   💾 Đã lưu threshold={threshold:.4f} → {json_path}")


# def load_threshold_from_json(project_root: Path) -> float | None:
#     """Đọc threshold từ project_info.json. Trả về None nếu chưa có."""
#     json_path = project_root / PROJECT_INFO_FILENAME
#     if not json_path.exists():
#         return None
#     with open(json_path, "r", encoding="utf-8") as f:
#         info = json.load(f)
#     return info.get("settings", {}).get("threshold", None)


# ─────────────────────────────────────────────
# TÍNH SCORE CHO MỘT ẢNH (không build anomaly map)
# ─────────────────────────────────────────────
def _compute_score(img_t: torch.Tensor,
                   model_engine,
                   bank: torch.Tensor) -> float:
    """
    Tính PatchCore anomaly score cho một ảnh tensor.
    Dùng chung logic distance với run_inspection_batch.
    """
    with torch.inference_mode():
        feat     = model_engine(img_t)
        feat_f32 = feat.float()
        bank_f32 = bank.float()

        a_sq    = (feat_f32 ** 2).sum(dim=1, keepdim=True)
        b_sq    = (bank_f32 ** 2).sum(dim=1).unsqueeze(0)
        dist_sq = (a_sq + b_sq - 2.0 * torch.mm(feat_f32, bank_f32.t())).clamp(min=1e-6)

        patch_dist = dist_sq.min(dim=1).values.sqrt()   # (P,)
        score      = patch_dist.max().item()             # anomaly score

    return score


# ─────────────────────────────────────────────
# TÁCH TRAIN / VAL NGẪU NHIÊN
# ─────────────────────────────────────────────
def _split_train_val(image_files: list, train_ratio: float, seed: int = 42):
    """
    Chia danh sách ảnh thành (train_files, val_files) theo tỉ lệ.

    Args:
        image_files: Danh sách Path ảnh.
        train_ratio: Tỉ lệ dùng để build bank, VD 0.8 = 80%.
        seed:        Random seed để tái lập kết quả.

    Returns:
        (train_files, val_files)
    """
    import random
    rng = random.Random(seed)
    shuffled = image_files[:]
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]


# ─────────────────────────────────────────────
# TÍNH NGƯỠNG TỐI ƯU TỰ ĐỘNG
# ─────────────────────────────────────────────
def auto_threshold(bank: torch.Tensor,
                   val_files: list,
                   model_engine,
                   project_root: Path,
                   safety_factor: float = 1.1) -> float:
    """
    Chạy inference trên val_files (ảnh KHÔNG dùng để build bank),
    lấy score cao nhất × safety_factor làm ngưỡng,
    ghi vào project_root/project_info.json → settings.threshold.

    Args:
        bank:          Memory Bank đã build (tensor trên device).
        val_files:     Danh sách Path ảnh validation (20% còn lại).
        model_engine:  Engine đã khởi tạo (dùng lại, không tạo lại).
        project_root:  Thư mục gốc project chứa project_info.json.
        safety_factor: Hệ số an toàn (mặc định 1.1 = +10%).
    """
    print(f"\n📏 Tính ngưỡng trên {len(val_files)} ảnh validation ...")

    if not val_files:
        raise ValueError("val_files rỗng, không thể tính ngưỡng.")

    scores = []
    for pth in val_files:
        img_t, _ = preprocess_any_size(pth)
        s = _compute_score(img_t, model_engine, bank)
        scores.append(s)
        print(f"   {pth.name:<30}  score = {s:.4f}")

    max_score  = max(scores)
    mean_score = sum(scores) / len(scores)
    threshold  = max_score * safety_factor

    print(f"\n   Val samples : {len(scores)} ảnh")
    print(f"   Score TB    : {mean_score:.4f}")
    print(f"   Score MAX   : {max_score:.4f}")
    print(f"   Safety      : ×{safety_factor}")
    print(f"   🎯 Ngưỡng   : {threshold:.4f}")

    _save_threshold_to_json(project_root, threshold)
    return threshold


# ─────────────────────────────────────────────
# XÂY DỰNG DYNAMIC BANK (CORESET) + AUTO THRESHOLD
# ─────────────────────────────────────────────
def build_bank(goods_dir: Path,
               output_dir: Path | None = None,
               project_root: Path | None = None,
               train_ratio: float = 0.8,
               safety_factor: float = 1.1,
               seed: int = 42):
    """
    Chia ảnh good theo tỉ lệ train/val, build Memory Bank từ phần train,
    tính ngưỡng tối ưu từ phần val, ghi threshold vào project_info.json.

    Args:
        goods_dir:     Thư mục chứa ảnh good.
                       VD: projects/ten_project/goods/
        output_dir:    Thư mục lưu memory_bank.pt.
                       Mặc định = goods_dir/../memory_bank/
        project_root:  Thư mục gốc project chứa project_info.json.
                       Mặc định = goods_dir.parent
        train_ratio:   Tỉ lệ ảnh dùng để build bank (mặc định 0.8 = 80%).
                       Phần còn lại (20%) dùng để tính ngưỡng.
        safety_factor: Hệ số an toàn cho ngưỡng (mặc định 1.1).
        seed:          Random seed để tái lập kết quả split (mặc định 42).

    Returns:
        (bank, threshold): tensor bank trên device, float threshold.
    """
    goods_dir = Path(goods_dir)

    # ── Đường dẫn output ─────────────────────────────────────────────────── #
    if output_dir is None:
        output_dir = goods_dir.parent / "memory_bank"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if project_root is None:
        project_root = goods_dir.parent
    project_root = Path(project_root)

    bank_path = output_dir / MEMORY_BANK_FILENAME

    # ── Lấy toàn bộ ảnh rồi chia train/val ──────────────────────────────── #
    all_files = sorted([
        f for f in goods_dir.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {goods_dir}")

    train_files, val_files = _split_train_val(all_files, train_ratio, seed)

    print(f"── Đang xây dựng Bank từ tập huấn luyện (Good images) ──")
    print(f"   📂 Nguồn   : {goods_dir}")
    print(f"   📊 Tổng    : {len(all_files)} ảnh  "
          f"→  train={len(train_files)}  val={len(val_files)}  "
          f"(ratio={train_ratio:.0%})")
    print(f"   💾 Lưu tại : {bank_path}")

    # ── Khởi tạo model (dùng chung cho cả build lẫn threshold) ───────────── #
    model_engine = get_engine(STD_W, STD_H)

    # ── Build features từ train_files ────────────────────────────────────── #
    all_feats = []
    with torch.inference_mode():
        for pth in train_files:
            img_t, _ = preprocess_any_size(pth)
            feat = model_engine(img_t)
            all_feats.append(feat.cpu())

    full_bank = torch.cat(all_feats, dim=0)

    # Dynamic Coreset: lấy mẫu hệ thống để bao phủ toàn bộ đặc trưng
    if full_bank.shape[0] > TARGET_BANK_SIZE:
        indices = torch.linspace(
            0, full_bank.shape[0] - 1, steps=TARGET_BANK_SIZE
        ).long()
        bank = full_bank[indices]
    else:
        bank = full_bank

    torch.save(bank, bank_path)
    print(f"✅ Memory Bank đã lưu: {bank_path}  (shape={tuple(bank.shape)})")

    # ── Tính ngưỡng từ val_files + ghi vào project_info.json ─────────────── #
    bank_on_device = bank.to(device).contiguous()
    threshold = auto_threshold(
        bank          = bank_on_device,
        val_files     = val_files,
        model_engine  = model_engine,
        project_root  = project_root,
        safety_factor = safety_factor,
    )

    return bank_on_device


# # ─────────────────────────────────────────────
# # LOAD BANK + THRESHOLD
# # ─────────────────────────────────────────────
# def load_bank(project_root: Path):
#     """
#     Load Memory Bank (memory_bank/memory_bank.pt)
#     và threshold (project_info.json → settings.threshold).

#     Args:
#         project_root: Thư mục gốc project.
#                       VD: projects/ten_project/

#     Returns:
#         (bank, threshold): threshold là float hoặc None nếu chưa có.
#     """
#     project_root = Path(project_root)
#     bank_path    = project_root / "memory_bank" / MEMORY_BANK_FILENAME

#     if not bank_path.exists():
#         raise FileNotFoundError(f"Không tìm thấy Memory Bank: {bank_path}")

#     bank = torch.load(bank_path, map_location=device).contiguous()
#     print(f"✅ Loaded Memory Bank : {bank_path}  (shape={tuple(bank.shape)})")

#     threshold = load_threshold_from_json(project_root)
#     if threshold is not None:
#         print(f"✅ Loaded threshold   : {threshold:.4f}")
#     else:
#         print(f"⚠️  Chưa có threshold trong project_info.json")

#     return bank, threshold