# import json
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from training.engine import get_engine
# from core.path_manager import BASE_DIR


# MEMORY_BANK_FILENAME  = "memory_bank.pt"
# PROJECT_INFO_FILENAME = "project_info.json"

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
# # GHI threshold vào project_info.json
# # ─────────────────────────────────────────────
# def _save_threshold_to_json(project_root: Path, threshold: float):
#     """Ghi threshold vào settings trong project_info.json."""
#     json_path = project_root / PROJECT_INFO_FILENAME
#     if not json_path.exists():
#         print(f"⚠️  Không tìm thấy {json_path}, bỏ qua lưu threshold.")
#         return

#     with open(json_path, "r", encoding="utf-8") as f:
#         info = json.load(f)

#     info.setdefault("settings", {})["threshold"] = round(float(threshold), 2)

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(info, f, indent=4, ensure_ascii=False)

#     print(f"   💾 Đã lưu threshold={threshold:.4f} → {json_path}")


# # def load_threshold_from_json(project_root: Path) -> float | None:
# #     """Đọc threshold từ project_info.json. Trả về None nếu chưa có."""
# #     json_path = project_root / PROJECT_INFO_FILENAME
# #     if not json_path.exists():
# #         return None
# #     with open(json_path, "r", encoding="utf-8") as f:
# #         info = json.load(f)
# #     return info.get("settings", {}).get("threshold", None)


# # ─────────────────────────────────────────────
# # TÍNH SCORE CHO MỘT ẢNH (không build anomaly map)
# # ─────────────────────────────────────────────
# def _compute_score(img_t: torch.Tensor,
#                    model_engine,
#                    bank: torch.Tensor) -> float:
#     """
#     Tính PatchCore anomaly score cho một ảnh tensor.
#     Dùng chung logic distance với run_inspection_batch.
#     """
#     with torch.inference_mode():
#         feat     = model_engine(img_t)
#         feat_f32 = feat.float()
#         bank_f32 = bank.float()

#         a_sq    = (feat_f32 ** 2).sum(dim=1, keepdim=True)
#         b_sq    = (bank_f32 ** 2).sum(dim=1).unsqueeze(0)
#         dist_sq = (a_sq + b_sq - 2.0 * torch.mm(feat_f32, bank_f32.t())).clamp(min=1e-6)

#         patch_dist = dist_sq.min(dim=1).values.sqrt()   # (P,)
#         score      = patch_dist.max().item()             # anomaly score

#     return score


# # ─────────────────────────────────────────────
# # TÁCH TRAIN / VAL NGẪU NHIÊN
# # ─────────────────────────────────────────────
# def _split_train_val(image_files: list, train_ratio: float, seed: int = 42):
#     """
#     Chia danh sách ảnh thành (train_files, val_files) theo tỉ lệ.

#     Args:
#         image_files: Danh sách Path ảnh.
#         train_ratio: Tỉ lệ dùng để build bank, VD 0.8 = 80%.
#         seed:        Random seed để tái lập kết quả.

#     Returns:
#         (train_files, val_files)
#     """
#     import random
#     rng = random.Random(seed)
#     shuffled = image_files[:]
#     rng.shuffle(shuffled)

#     n_train = max(1, int(len(shuffled) * train_ratio))
#     return shuffled[:n_train], shuffled[n_train:]


# # ─────────────────────────────────────────────
# # TÍNH NGƯỠNG TỐI ƯU TỰ ĐỘNG
# # ─────────────────────────────────────────────
# def auto_threshold(bank: torch.Tensor,
#                    val_files: list,
#                    model_engine,
#                    project_root: Path,
#                    safety_factor: float = 1.1) -> float:
#     """
#     Chạy inference trên val_files (ảnh KHÔNG dùng để build bank),
#     lấy score cao nhất × safety_factor làm ngưỡng,
#     ghi vào project_root/project_info.json → settings.threshold.

#     Args:
#         bank:          Memory Bank đã build (tensor trên device).
#         val_files:     Danh sách Path ảnh validation (20% còn lại).
#         model_engine:  Engine đã khởi tạo (dùng lại, không tạo lại).
#         project_root:  Thư mục gốc project chứa project_info.json.
#         safety_factor: Hệ số an toàn (mặc định 1.1 = +10%).
#     """
#     print(f"\n📏 Tính ngưỡng trên {len(val_files)} ảnh validation ...")

#     if not val_files:
#         raise ValueError("val_files rỗng, không thể tính ngưỡng.")

#     scores = []
#     for pth in val_files:
#         img_t, _ = preprocess_any_size(pth)
#         s = _compute_score(img_t, model_engine, bank)
#         scores.append(s)
#         print(f"   {pth.name:<30}  score = {s:.4f}")

#     max_score  = max(scores)
#     mean_score = sum(scores) / len(scores)
#     threshold  = max_score * safety_factor

#     print(f"\n   Val samples : {len(scores)} ảnh")
#     print(f"   Score TB    : {mean_score:.4f}")
#     print(f"   Score MAX   : {max_score:.4f}")
#     print(f"   Safety      : ×{safety_factor}")
#     print(f"   🎯 Ngưỡng   : {threshold:.4f}")

#     _save_threshold_to_json(project_root, threshold)
#     return threshold


# # ─────────────────────────────────────────────
# # XÂY DỰNG DYNAMIC BANK (CORESET) + AUTO THRESHOLD
# # ─────────────────────────────────────────────
# def build_bank(goods_dir: Path,
#                output_dir: Path | None = None,
#                project_root: Path | None = None,
#                train_ratio: float = 0.8,
#                safety_factor: float = 1,
#                seed: int = 42):
#     """
#     Chia ảnh good theo tỉ lệ train/val, build Memory Bank từ phần train,
#     tính ngưỡng tối ưu từ phần val, ghi threshold vào project_info.json.

#     Args:
#         goods_dir:     Thư mục chứa ảnh good.
#                        VD: projects/ten_project/goods/
#         output_dir:    Thư mục lưu memory_bank.pt.
#                        Mặc định = goods_dir/../memory_bank/
#         project_root:  Thư mục gốc project chứa project_info.json.
#                        Mặc định = goods_dir.parent
#         train_ratio:   Tỉ lệ ảnh dùng để build bank (mặc định 0.8 = 80%).
#                        Phần còn lại (20%) dùng để tính ngưỡng.
#         safety_factor: Hệ số an toàn cho ngưỡng (mặc định 1.1).
#         seed:          Random seed để tái lập kết quả split (mặc định 42).

#     Returns:
#         (bank, threshold): tensor bank trên device, float threshold.
#     """
#     goods_dir = Path(goods_dir)

#     # ── Đường dẫn output ─────────────────────────────────────────────────── #
#     if output_dir is None:
#         output_dir = goods_dir.parent / "memory_bank"
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if project_root is None:
#         project_root = goods_dir.parent
#     project_root = Path(project_root)

#     bank_path = output_dir / MEMORY_BANK_FILENAME

#     # ── Lấy toàn bộ ảnh rồi chia train/val ──────────────────────────────── #
#     all_files = sorted([
#         f for f in goods_dir.iterdir()
#         if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
#     ])
#     if not all_files:
#         raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {goods_dir}")

#     train_files, val_files = _split_train_val(all_files, train_ratio, seed)

#     print(f"── Đang xây dựng Bank từ tập huấn luyện (Good images) ──")
#     print(f"   📂 Nguồn   : {goods_dir}")
#     print(f"   📊 Tổng    : {len(all_files)} ảnh  "
#           f"→  train={len(train_files)}  val={len(val_files)}  "
#           f"(ratio={train_ratio:.0%})")
#     print(f"   💾 Lưu tại : {bank_path}")

#     # ── Khởi tạo model (dùng chung cho cả build lẫn threshold) ───────────── #
#     model_engine = get_engine(STD_W, STD_H)

#     # ── Build features từ train_files ────────────────────────────────────── #
#     all_feats = []
#     with torch.inference_mode():
#         for pth in train_files:
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

#     # ── Tính ngưỡng từ val_files + ghi vào project_info.json ─────────────── #
#     bank_on_device = bank.to(device).contiguous()
#     threshold = auto_threshold(
#         bank          = bank_on_device,
#         val_files     = val_files,
#         model_engine  = model_engine,
#         project_root  = project_root,
#         safety_factor = safety_factor,
#     )

#     return bank_on_device


# # # ─────────────────────────────────────────────
# # # LOAD BANK + THRESHOLD
# # # ─────────────────────────────────────────────
# # def load_bank(project_root: Path):
# #     """
# #     Load Memory Bank (memory_bank/memory_bank.pt)
# #     và threshold (project_info.json → settings.threshold).

# #     Args:
# #         project_root: Thư mục gốc project.
# #                       VD: projects/ten_project/

# #     Returns:
# #         (bank, threshold): threshold là float hoặc None nếu chưa có.
# #     """
# #     project_root = Path(project_root)
# #     bank_path    = project_root / "memory_bank" / MEMORY_BANK_FILENAME

# #     if not bank_path.exists():
# #         raise FileNotFoundError(f"Không tìm thấy Memory Bank: {bank_path}")

# #     bank = torch.load(bank_path, map_location=device).contiguous()
# #     print(f"✅ Loaded Memory Bank : {bank_path}  (shape={tuple(bank.shape)})")

# #     threshold = load_threshold_from_json(project_root)
# #     if threshold is not None:
# #         print(f"✅ Loaded threshold   : {threshold:.4f}")
# #     else:
# #         print(f"⚠️  Chưa có threshold trong project_info.json")

# #     return bank, threshold



import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from training.engine import get_engine
from core.path_manager import BASE_DIR
import cv2

MEMORY_BANK_FILENAME  = "memory_bank.pt"
PROJECT_INFO_FILENAME = "project_info.json"
U2NET_ENGINE_PATH     = str(Path(BASE_DIR) / "models/remove_bg/u2netp.trt")

device = torch.device("cuda")
dtype  = torch.float16

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

STD_W, STD_H     = 224, 224
TARGET_BANK_SIZE = 8000
FEAT_H = FEAT_W  = 28   # WideResNet50 → 28×28 patches


# =========================================================================== #
#  TIỀN XỬ LÝ                                                                 #
# =========================================================================== #
def preprocess_any_size(path):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img_res = img.resize((STD_W, STD_H), Image.LANCZOS)
    img_t = (
        torch.from_numpy(np.array(img_res))
        .to(device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=dtype)
    )
    return (img_t / 255.0 - MEAN) / STD, (orig_w, orig_h)


# =========================================================================== #
#  U2NET SEGMENTOR (lazy init, dùng chung 1 lần cho toàn bộ val)              #
# =========================================================================== #
def _init_segmentor():
    """
    Khởi tạo U2NetSegmentor nếu engine tồn tại.
    Trả về instance hoặc None nếu không có engine.
    Gọi attach() sau khi trả về trước khi dùng.
    """
    if not Path(U2NET_ENGINE_PATH).exists():
        print(f"[AutoThreshold] ⚠️  Không tìm thấy {U2NET_ENGINE_PATH} "
              f"— tính threshold không có mask")
        return None
    try:
        from core.u2net_segmentor import U2NetSegmentor
        seg = U2NetSegmentor(U2NET_ENGINE_PATH)
        print(f"[AutoThreshold] ✅ U2Net loaded cho threshold calculation")
        return seg
    except Exception as e:
        print(f"[AutoThreshold] ⚠️  U2Net load thất bại: {e} — không có mask")
        return None


def _get_mask_for_image(seg, img_path: Path) -> np.ndarray | None:
    """
    Đọc ảnh BGR → U2Net mask uint8 [0,255].
    seg phải đã attach() trước khi gọi hàm này.
    """
    if seg is None:
        return None
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        return seg.get_mask(img_bgr)
    except Exception as e:
        print(f"[AutoThreshold] ⚠️  mask lỗi {img_path.name}: {e}")
        return None


# =========================================================================== #
#  TÍNH SCORE                                                                  #
# =========================================================================== #
def _compute_score(img_t: torch.Tensor,
                   model_engine,
                   bank: torch.Tensor,
                   object_mask: np.ndarray = None) -> float:
    """
    Tính PatchCore anomaly score cho một ảnh tensor.

    Nếu object_mask được truyền vào:
      - Resize mask về feature map (28×28)
      - Chỉ lấy max score của các patch NẰM TRONG mask
      → Loại bỏ noise từ background, threshold chính xác hơn

    Nếu object_mask=None: dùng max score toàn ảnh (behavior cũ).
    """
    with torch.inference_mode():
        feat     = model_engine(img_t)              # (784, C)
        feat_f32 = feat.float()
        bank_f32 = bank.float()

        a_sq    = (feat_f32 ** 2).sum(dim=1, keepdim=True)
        b_sq    = (bank_f32 ** 2).sum(dim=1).unsqueeze(0)
        dist_sq = (a_sq + b_sq - 2.0 * torch.mm(feat_f32, bank_f32.t())).clamp(min=1e-6)

        patch_dist = dist_sq.min(dim=1).values.sqrt()   # (784,) — score từng patch

    patch_dist_np = patch_dist.cpu().float().numpy().reshape(FEAT_H, FEAT_W)

    if object_mask is not None:
        # Resize mask → feature map size (28×28)
        mask_feat      = cv2.resize(object_mask, (FEAT_W, FEAT_H),
                                    interpolation=cv2.INTER_NEAREST)
        mask_feat_bool = mask_feat > 127
        scores_in_mask = patch_dist_np[mask_feat_bool]

        if scores_in_mask.size > 0:
            score = float(scores_in_mask.max())
        else:
            # Mask quá nhỏ → fallback toàn ảnh
            score = float(patch_dist_np.max())
            print(f"[AutoThreshold] ⚠️  Mask quá nhỏ → dùng score toàn ảnh")
    else:
        score = float(patch_dist_np.max())

    return score


# =========================================================================== #
#  TÁCH TRAIN / VAL                                                            #
# =========================================================================== #
def _split_train_val(image_files: list, train_ratio: float, seed: int = 42):
    import random
    rng      = random.Random(seed)
    shuffled = image_files[:]
    rng.shuffle(shuffled)
    n_train  = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]


# =========================================================================== #
#  LƯU THRESHOLD                                                               #
# =========================================================================== #
def _save_threshold_to_json(project_root: Path, threshold: float):
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


# =========================================================================== #
#  AUTO THRESHOLD — dùng U2Net mask khi tính score val                         #
# =========================================================================== #
def auto_threshold(bank: torch.Tensor,
                   val_files: list,
                   model_engine,
                   project_root: Path,
                   n_sigma: float = 3.0) -> float:
    """
    Chạy inference trên val_files, tính score **trong vùng mask U2Net**
    (nếu engine có sẵn), tính threshold theo phân phối chuẩn:

        threshold = mean(scores) + n_sigma × std(scores)

    Ý nghĩa:
      - n_sigma=2 → bao phủ ~97.7% good samples
      - n_sigma=3 → bao phủ ~99.7% good samples  (mặc định)
      - n_sigma=4 → bao phủ ~99.99% good samples

    Ưu điểm so với max × safety_factor:
      - Robust hơn với outlier (1 ảnh score cao không kéo threshold lên)
      - Có ý nghĩa thống kê rõ ràng
      - Score trong mask U2Net → loại bỏ background noise

    Args:
        bank      : Memory Bank tensor trên device
        val_files : list[Path] ảnh validation
        model_engine : backbone TRT engine
        project_root : thư mục chứa project_info.json
        n_sigma   : số sigma (mặc định 3.0)
    """
    if not val_files:
        raise ValueError("val_files rỗng, không thể tính ngưỡng.")

    print(f"\n📏 Tính ngưỡng trên {len(val_files)} ảnh validation ...")

    # ── Khởi tạo U2Net (dùng chung cho tất cả val ảnh) ───────────────────
    seg = _init_segmentor()
    use_mask = seg is not None

    if use_mask:
        seg.attach()
        print("[AutoThreshold] 🎭 Tính score trong vùng mask U2Net")
    else:
        print("[AutoThreshold] ⬜ Tính score toàn ảnh (không có mask)")

    # ── Phase 1: U2Net lấy tất cả masks trước — detach() trước khi PatchCore ──
    # Tách hoàn toàn 2 TRT engine, tránh Cask convolution conflict.
    masks = [None] * len(val_files)
    if use_mask:
        print("[AutoThreshold] Phase 1/2: U2Net masks ...")
        try:
            for i, pth in enumerate(val_files):
                masks[i] = _get_mask_for_image(seg, pth)
        finally:
            # Detach U2Net context TRƯỚC khi PatchCore chạy
            try:
                seg.detach()
            except Exception:
                pass
            use_mask = False   # đánh dấu đã detach, không detach lại ở finally
        torch.cuda.synchronize()   # flush hết pycuda ops trước khi PyTorch dùng GPU
        print("[AutoThreshold] Phase 1/2: U2Net done ✓")

    # ── Phase 2: PatchCore tính score (U2Net đã detach hoàn toàn) ────────────
    print("[AutoThreshold] Phase 2/2: PatchCore scoring ...")
    scores = []
    try:
        for i, pth in enumerate(val_files):
            img_t, _ = preprocess_any_size(pth)
            s = _compute_score(img_t, model_engine, bank, object_mask=masks[i])
            scores.append(s)
            mask_str = "masked" if masks[i] is not None else "no-mask"
            print(f"   {pth.name:<35}  score={s:.4f}  [{mask_str}]")

    finally:
        if use_mask and seg is not None:   # fallback detach nếu phase 1 bị lỗi giữa chừng
            try:
                seg.detach()
            except Exception:
                pass

    scores_np  = np.array(scores, dtype=np.float32)
    mean_score = float(scores_np.mean())
    std_score  = float(scores_np.std())
    max_score  = float(scores_np.max())
    min_score  = float(scores_np.min())
    threshold  = mean_score + n_sigma * std_score

    print(f"\n   Val samples : {len(scores)} ảnh")
    print(f"   Mask        : {'U2Net ✓' if masks[0] is not None or any(m is not None for m in masks) else 'không có ✗'}")
    print(f"   Score min   : {min_score:.4f}")
    print(f"   Score max   : {max_score:.4f}")
    print(f"   Score mean  : {mean_score:.4f}")
    print(f"   Score std   : {std_score:.4f}")
    print(f"   n_sigma     : {n_sigma}")
    print(f"   🎯 Ngưỡng   : {mean_score:.4f} + {n_sigma} × {std_score:.4f} = {threshold:.4f}")

    _save_threshold_to_json(project_root, threshold)
    return threshold


# =========================================================================== #
#  BUILD BANK + AUTO THRESHOLD                                                 #
# =========================================================================== #
def build_bank(goods_dir: Path,
               output_dir: Path | None = None,
               project_root: Path | None = None,
               train_ratio: float = 0.8,
               n_sigma: float = 3.0,
               seed: int = 42):
    """
    Chia ảnh good theo tỉ lệ train/val, build Memory Bank từ phần train,
    tính ngưỡng theo phân phối chuẩn: mean + n_sigma × std từ val scores.

    Args:
        n_sigma : số sigma cho threshold (mặc định 3.0 → bao phủ ~99.7%)
    """
    goods_dir = Path(goods_dir)

    if output_dir is None:
        output_dir = goods_dir.parent / "memory_bank"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if project_root is None:
        project_root = goods_dir.parent
    project_root = Path(project_root)

    bank_path = output_dir / MEMORY_BANK_FILENAME

    # ── Lấy ảnh + chia train/val ─────────────────────────────────────────
    all_files = sorted([
        f for f in goods_dir.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {goods_dir}")

    train_files, val_files = _split_train_val(all_files, train_ratio, seed)

    print(f"── Xây dựng Memory Bank ──")
    print(f"   📂 Nguồn   : {goods_dir}")
    print(f"   📊 Tổng    : {len(all_files)} ảnh  "
          f"→  train={len(train_files)}  val={len(val_files)}  "
          f"(ratio={train_ratio:.0%})")
    print(f"   💾 Lưu tại : {bank_path}")

    # ── Khởi tạo model (dùng chung build + threshold) ────────────────────
    model_engine = get_engine(STD_W, STD_H)

    # ── Build features từ train_files ────────────────────────────────────
    all_feats = []
    with torch.inference_mode():
        for pth in train_files:
            img_t, _ = preprocess_any_size(pth)
            feat = model_engine(img_t)
            all_feats.append(feat.cpu())

    full_bank = torch.cat(all_feats, dim=0)

    # Coreset sampling
    if full_bank.shape[0] > TARGET_BANK_SIZE:
        indices = torch.linspace(
            0, full_bank.shape[0] - 1, steps=TARGET_BANK_SIZE
        ).long()
        bank = full_bank[indices]
    else:
        bank = full_bank

    torch.save(bank, bank_path)
    print(f"✅ Memory Bank đã lưu: {bank_path}  (shape={tuple(bank.shape)})")

    # ── Tính threshold từ val + U2Net mask ───────────────────────────────
    bank_on_device = bank.to(device).contiguous()
    threshold = auto_threshold(
        bank         = bank_on_device,
        val_files    = val_files,
        model_engine = model_engine,
        project_root = project_root,
        n_sigma      = n_sigma,
    )

    return bank_on_device


POSITION_BANK_FILENAMES = {
    "body": "body_memory_bank.pt",
    "cap": "cap_memory_bank.pt",
}


def _save_threshold_to_json(project_root: Path, threshold: float, model_name: str = "body"):
    json_path = project_root / PROJECT_INFO_FILENAME
    if not json_path.exists():
        print(f"[BuildBank] Missing {json_path}, skip save threshold.")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    settings = info.setdefault("settings", {})
    settings.setdefault("thresholds", {})[model_name] = round(float(threshold), 2)
    if model_name == "body":
        settings["threshold"] = round(float(threshold), 2)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f"[BuildBank] Saved threshold[{model_name}]={threshold:.4f}")


def auto_threshold(bank: torch.Tensor,
                   val_files: list,
                   model_engine,
                   project_root: Path,
                   n_sigma: float = 3.0,
                   model_name: str = "body") -> float:
    if not val_files:
        raise ValueError(f"val_files for {model_name} is empty")

    print(f"\n[BuildBank] Auto threshold for {model_name} on {len(val_files)} images")
    seg = _init_segmentor()
    masks = [None] * len(val_files)
    if seg is not None:
        seg.attach()
        try:
            for i, pth in enumerate(val_files):
                masks[i] = _get_mask_for_image(seg, pth)
        finally:
            try:
                seg.detach()
            except Exception:
                pass
        torch.cuda.synchronize()

    scores = []
    for i, pth in enumerate(val_files):
        img_t, _ = preprocess_any_size(pth)
        score = _compute_score(img_t, model_engine, bank, object_mask=masks[i])
        scores.append(score)
        print(f"   {pth.name:<35} score={score:.4f}")

    scores_np = np.array(scores, dtype=np.float32)
    mean_score = float(scores_np.mean())
    std_score = float(scores_np.std())
    threshold = mean_score + n_sigma * std_score
    print(f"[BuildBank] threshold[{model_name}] = {mean_score:.4f} + {n_sigma} * {std_score:.4f} = {threshold:.4f}")
    _save_threshold_to_json(project_root, threshold, model_name=model_name)
    return threshold


def _group_good_images_by_position(goods_dir: Path) -> dict[str, list[Path]]:
    grouped = {"body": [], "cap": []}
    for path in sorted(goods_dir.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        lower_name = path.name.lower()
        if lower_name.startswith("body_"):
            grouped["body"].append(path)
        elif lower_name.startswith("cap_"):
            grouped["cap"].append(path)
    return grouped


def _build_single_bank(train_files: list[Path], model_engine, bank_path: Path) -> torch.Tensor:
    all_feats = []
    with torch.inference_mode():
        for pth in train_files:
            img_t, _ = preprocess_any_size(pth)
            feat = model_engine(img_t)
            all_feats.append(feat.cpu())

    full_bank = torch.cat(all_feats, dim=0)
    if full_bank.shape[0] > TARGET_BANK_SIZE:
        indices = torch.linspace(0, full_bank.shape[0] - 1, steps=TARGET_BANK_SIZE).long()
        bank = full_bank[indices]
    else:
        bank = full_bank
    torch.save(bank, bank_path)
    print(f"[BuildBank] Saved {bank_path.name} shape={tuple(bank.shape)}")
    return bank


def build_bank(goods_dir: Path,
               output_dir: Path | None = None,
               project_root: Path | None = None,
               train_ratio: float = 0.8,
               n_sigma: float = 3.0,
               seed: int = 42):
    goods_dir = Path(goods_dir)
    if output_dir is None:
        output_dir = goods_dir.parent / "memory_bank"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if project_root is None:
        project_root = goods_dir.parent
    project_root = Path(project_root)

    grouped_files = _group_good_images_by_position(goods_dir)
    if not any(grouped_files.values()):
        raise FileNotFoundError(f"Khong tim thay anh body_/cap_ trong: {goods_dir}")

    model_engine = get_engine(STD_W, STD_H)
    built_banks = {}

    for model_name, image_files in grouped_files.items():
        if not image_files:
            print(f"[BuildBank] Skip {model_name}: no images")
            continue

        train_files, val_files = _split_train_val(image_files, train_ratio, seed)
        bank_path = output_dir / POSITION_BANK_FILENAMES[model_name]
        print(f"[BuildBank] Build {model_name}: total={len(image_files)} train={len(train_files)} val={len(val_files)}")
        bank = _build_single_bank(train_files, model_engine, bank_path)
        bank_on_device = bank.to(device).contiguous()
        auto_threshold(
            bank=bank_on_device,
            val_files=val_files,
            model_engine=model_engine,
            project_root=project_root,
            n_sigma=n_sigma,
            model_name=model_name,
        )
        built_banks[model_name] = bank_on_device

    return built_banks
