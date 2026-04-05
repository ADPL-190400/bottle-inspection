# """
# Tách vật thể khỏi nền công nghiệp
# Input : background.jpg (ảnh nền), object.jpg (ảnh có vật)
# Output: result.png (ảnh đã tách, nền giữ nguyên, vật thể rõ ràng)

# Yêu cầu:
#     pip install opencv-python numpy
# """

# import cv2
# import numpy as np
# import sys
# import os

# # ─────────────────────────────────────────────
# # THAM SỐ ĐIỀU CHỈNH
# # ─────────────────────────────────────────────
# DIFF_THRESH      = 20      # ngưỡng diff (0-255). Tăng nếu hay bị nhiễu, giảm để nhạy hơn
# MORPH_CLOSE_SIZE = 25      # kích thước kernel đóng lỗ hổng trong mask
# MORPH_OPEN_SIZE  = 25       # kích thước kernel loại nhiễu nhỏ
# MIN_CONTOUR_AREA = 500     # bỏ qua vùng nhỏ hơn N pixel (nhiễu)
# BRIGHTNESS_COMP  = False   # bật bù sáng giữa 2 ảnh
# # ─────────────────────────────────────────────


# def load_image(path: str) -> np.ndarray:
#     """Đọc ảnh, raise lỗi rõ ràng nếu không tìm thấy."""
#     img = cv2.imread(path)
#     if img is None:
#         raise FileNotFoundError(f"Không đọc được ảnh: {path}")
#     return img


# def match_brightness(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
#     """
#     Bù sáng nhẹ: dịch chuyển mean brightness của src về gần ref.
#     Chỉ thay đổi ảnh vật (src), nền (ref) giữ nguyên.
#     """
#     src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
#     ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

#     # Chỉ bù kênh L (độ sáng)
#     diff_L = ref_lab[:, :, 0].mean() - src_lab[:, :, 0].mean()
#     src_lab[:, :, 0] = np.clip(src_lab[:, :, 0] + diff_L * 0.5, 0, 255)

#     result = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
#     return result


# def build_mask(bg: np.ndarray, obj: np.ndarray) -> np.ndarray:
#     """
#     Tạo mask nhị phân: 255 = vật thể, 0 = nền.
#     Sử dụng diff trên ảnh grayscale + morphology.
#     """
#     # --- 1. Resize nếu khác kích thước ---
#     if bg.shape != obj.shape:
#         obj = cv2.resize(obj, (bg.shape[1], bg.shape[0]))

#     # --- 2. Bù sáng ---
#     if BRIGHTNESS_COMP:
#         obj_adj = match_brightness(obj, bg)
#     else:
#         obj_adj = obj.copy()

#     # --- 3. Diff grayscale ---
#     bg_gray  = cv2.cvtColor(bg,      cv2.COLOR_BGR2GRAY)
#     obj_gray = cv2.cvtColor(obj_adj, cv2.COLOR_BGR2GRAY)

#     diff = cv2.absdiff(bg_gray, obj_gray)

#     # --- 4. Threshold ---
#     _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

#     # --- 5. Morphology: đóng lỗ → loại nhiễu ---
#     kernel_close = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE, (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
#     kernel_open  = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE, (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))

#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

#     # --- 6. Loại contour nhỏ ---
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     clean_mask  = np.zeros_like(mask)
#     for cnt in contours:
#         if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
#             cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

#     # --- 7. Làm mềm biên (feathering nhẹ) ---
#     soft_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)

#     return obj_adj, soft_mask


# def composite(bg: np.ndarray,
#               obj_adj: np.ndarray,
#               mask: np.ndarray) -> np.ndarray:
#     """
#     Ghép vật thể lên nền bằng alpha blending.
#     Nền gốc giữ nguyên ở vùng không có vật.
#     """
#     alpha = mask.astype(np.float32) / 255.0
#     alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)

#     bg_f   = bg.astype(np.float32)
#     obj_f  = obj_adj.astype(np.float32)

#     result = obj_f * alpha_3ch + bg_f * (1.0 - alpha_3ch)
#     return np.clip(result, 0, 255).astype(np.uint8)


# def save_debug(mask: np.ndarray, out_dir: str):
#     """Lưu mask để kiểm tra."""
#     path = os.path.join(out_dir, "debug_mask.png")
#     cv2.imwrite(path, mask)
#     print(f"  [debug] mask → {path}")


# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# def main():
#     # Cho phép truyền tên file qua CLI: python tach_vat.py bg.jpg obj.jpg [out.png]
#     bg_path  = sys.argv[1] if len(sys.argv) > 1 else "models/bg2.jpg"
#     obj_path = sys.argv[2] if len(sys.argv) > 2 else "models/3.jpg"
#     out_path = sys.argv[3] if len(sys.argv) > 3 else "result.png"

#     print(f"[1/4] Đọc ảnh nền  : {bg_path}")
#     bg  = load_image(bg_path)

#     print(f"[2/4] Đọc ảnh vật  : {obj_path}")
#     obj = load_image(obj_path)

#     print("[3/4] Tạo mask & bù sáng ...")
#     obj_adj, mask = build_mask(bg, obj)

#     # Lưu mask debug cạnh output
#     out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
#     save_debug(mask, out_dir)

#     print("[4/4] Ghép ảnh & lưu kết quả ...")
#     # result = composite(bg, obj_adj, mask)
#     result = mask
#     cv2.imwrite(out_path, result)

#     print(f"\n✅ Xong! Kết quả: {out_path}")
#     print(f"   Kích thước    : {result.shape[1]}x{result.shape[0]} px")

#     # Thống kê vùng vật thể
#     area_ratio = (mask > 127).sum() / mask.size * 100
#     print(f"   Vùng vật thể  : {area_ratio:.1f}% diện tích ảnh")


# if __name__ == "__main__":
#     main()


"""
Tách vật thể dùng cv2.createBackgroundSubtractorKNN()
Input : background.jpg (hoặc nhiều ảnh nền), object.jpg
Output: result.png, debug_mask.png

Cài đặt:
    pip install opencv-python numpy

Cách dùng:
    python tach_vat_knn.py
    python tach_vat_knn.py background.jpg object.jpg result.png

Ý tưởng:
    BackgroundSubtractorKNN học phân phối màu của nền qua nhiều frame.
    Ta "feed" ảnh nền nhiều lần (với augmentation nhẹ) để model học
    biến thiên tự nhiên của nền (nhiễu cảm biến, rung nhẹ, đổi sáng nhỏ).
    Sau đó apply ảnh có vật → KNN nhận ra pixel nào là foreground.
"""

import cv2
import numpy as np
import sys
import os

# ─────────────────────────────────────────────
# THAM SỐ
# ─────────────────────────────────────────────
# KNN params
KNN_HISTORY       = 1    # số frame lịch sử KNN ghi nhớ
KNN_DIST2THRESH   = 1000.0   # ngưỡng khoảng cách KNN (thấp=nhạy, cao=thô)
KNN_DETECT_SHADOW = True  # True: phát hiện bóng (127), False: bỏ qua

# Số lần feed ảnh nền (augmentation)
BG_FEED_TIMES     = 200      # feed nhiều lần để KNN học đủ

# Augmentation nhẹ khi feed nền (mô phỏng biến đổi ánh sáng/nhiễu thực tế)
AUG_BRIGHTNESS_RANGE = (-8, 8)    # thay đổi brightness ngẫu nhiên ±8
AUG_NOISE_STD        = 3          # Gaussian noise std
AUG_BLUR_PROB        = 0.3        # xác suất blur nhẹ mỗi frame

# Morphology
MORPH_CLOSE   = 5
MORPH_OPEN    = 25
MIN_AREA      = 3000

# Feather
FEATHER       = 5

# Bù sáng
BRIGHTNESS_COMP = False
# ─────────────────────────────────────────────


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được: {path}")
    return img


def match_brightness(src, ref):
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff    = ref_lab[:,:,0].mean() - src_lab[:,:,0].mean()
    src_lab[:,:,0] = np.clip(src_lab[:,:,0] + diff * 0.5, 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def augment_bg(img):
    """Augmentation nhẹ: brightness shift + Gaussian noise + blur ngẫu nhiên."""
    out = img.astype(np.float32)

    # Brightness shift
    delta = np.random.randint(*AUG_BRIGHTNESS_RANGE)
    out   = np.clip(out + delta, 0, 255)

    # Gaussian noise
    noise = np.random.normal(0, AUG_NOISE_STD, out.shape).astype(np.float32)
    out   = np.clip(out + noise, 0, 255)

    out = out.astype(np.uint8)

    # Blur nhẹ
    if np.random.rand() < AUG_BLUR_PROB:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


def build_knn_subtractor():
    """Tạo KNN subtractor với tham số đã cấu hình."""
    backSub = cv2.createBackgroundSubtractorKNN(
        history       = KNN_HISTORY,
        dist2Threshold= KNN_DIST2THRESH,
        detectShadows = KNN_DETECT_SHADOW
    )
    return backSub


def train_on_background(backSub, bg_img):
    """
    Feed ảnh nền nhiều lần với augmentation để KNN học
    phân phối màu và biến thiên tự nhiên của nền.
    """
    print(f"  Feeding nền vào KNN ({BG_FEED_TIMES} frames) ...")
    for i in range(BG_FEED_TIMES):
        frame = augment_bg(bg_img)
        # learningRate=-1: KNN tự quyết; 0: không học; >0: học nhanh
        # Những frame đầu học nhanh, sau đó chậm dần để ổn định
        lr = 0.1 if i < 20 else (0.05 if i < 50 else -1)
        backSub.apply(frame, learningRate=lr)

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{BG_FEED_TIMES}] done")


def apply_object_frame(backSub, obj_img):
    """
    Apply ảnh có vật vào KNN với learningRate=0
    (không cập nhật model — chỉ lấy foreground mask).
    """
    fg_mask = backSub.apply(obj_img, learningRate=0)

    # Nếu bật detectShadows, pixel bóng = 127; ta bỏ bóng
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    return fg_mask


def clean_mask(mask):
    """Morphology + lọc contour nhỏ + feather."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_CLOSE, MORPH_CLOSE))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_OPEN, MORPH_OPEN))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)

    # Lọc contour nhỏ
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)

    # Feather biên
    if FEATHER > 0:
        r     = FEATHER * 2 + 1
        clean = cv2.GaussianBlur(clean, (r, r), 0)

    return clean


def composite(bg, obj, mask):
    if bg.shape != obj.shape:
        obj = cv2.resize(obj, (bg.shape[1], bg.shape[0]))
    if mask.shape[:2] != bg.shape[:2]:
        mask = cv2.resize(mask, (bg.shape[1], bg.shape[0]))

    a  = mask.astype(np.float32) / 255.0
    a3 = np.stack([a] * 3, axis=-1)
    r  = obj.astype(np.float32) * a3 + bg.astype(np.float32) * (1 - a3)
    return np.clip(r, 0, 255).astype(np.uint8)


def save_debug(obj, raw_mask, clean_m, out_dir):
    """Lưu ảnh debug gồm: ảnh vật, raw KNN mask, clean mask."""
    h, w = obj.shape[:2]
    sc   = min(1.0, 400 / max(h, w))
    sw, sh = int(w * sc), int(h * sc)

    def rs(img): return cv2.resize(img, (sw, sh))
    def to3(m):  return cv2.cvtColor(rs(m), cv2.COLOR_GRAY2BGR)

    row = np.hstack([rs(obj), to3(raw_mask), to3(clean_m)])
    for i, lbl in enumerate(["Object", "KNN raw", "Clean mask"]):
        cv2.putText(row, lbl, (i * sw + 6, sh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1, cv2.LINE_AA)

    path = os.path.join(out_dir, "debug_knn.png")
    cv2.imwrite(path, row)
    print(f"  [debug] knn analysis → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    bg_path  = sys.argv[1] if len(sys.argv) > 1 else "models/bg1.jpg"
    obj_path = sys.argv[2] if len(sys.argv) > 2 else "models/3.jpg"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "result.png"

    print(f"[1/6] Đọc ảnh nền   : {bg_path}")
    bg  = load(bg_path)

    print(f"[2/6] Đọc ảnh vật   : {obj_path}")
    obj = load(obj_path)

    if bg.shape != obj.shape:
        obj = cv2.resize(obj, (bg.shape[1], bg.shape[0]))

    if BRIGHTNESS_COMP:
        obj = match_brightness(obj, bg)
        print("       Bù sáng: bật")

    print("[3/6] Khởi tạo KNN subtractor ...")
    backSub = build_knn_subtractor()
    print(f"  history={KNN_HISTORY}  dist2Threshold={KNN_DIST2THRESH}  detectShadows={KNN_DETECT_SHADOW}")

    print("[4/6] Train KNN trên ảnh nền ...")
    train_on_background(backSub, bg)

    print("[5/6] Apply ảnh có vật → foreground mask ...")
    raw_mask = apply_object_frame(backSub, obj)

    out_dir  = os.path.dirname(os.path.abspath(out_path)) or "."
    clean_m  = clean_mask(raw_mask)
    cv2.imwrite(os.path.join(out_dir, "debug_mask.png"), clean_m)
    save_debug(obj, raw_mask, clean_m, out_dir)

    print("[6/6] Ghép ảnh & lưu ...")
    result = composite(bg, obj, clean_m)
    cv2.imwrite(out_path, result)

    area = (clean_m > 127).sum() / clean_m.size * 100
    print(f"\n✅ Xong! → {out_path}")
    print(f"   Kích thước    : {result.shape[1]}×{result.shape[0]} px")
    print(f"   Vùng vật thể : {area:.1f}% ảnh")
    print("\n   Nếu mask chưa đẹp, thử điều chỉnh:")
    print("   KNN_DIST2THRESH  : giảm → nhạy hơn | tăng → ít nhiễu hơn")
    print("   BG_FEED_TIMES    : tăng → KNN học kỹ nền hơn")
    print("   AUG_BRIGHTNESS_RANGE: tăng range nếu ánh sáng biến đổi nhiều")


if __name__ == "__main__":
    main()