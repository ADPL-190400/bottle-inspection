"""
core/liquid_level.py
====================
Module phát hiện mức chất lỏng trong chai.

Workflow:
  Setup  : crop template từ ROI đã chọn → lưu water_template.png + liquid_config.json
           → chạy detect() trên ảnh mẫu → lấy fill_ratio làm baseline
           → ngưỡng OK = [baseline - tolerance, baseline + tolerance]
  Detect : template matching trên Canny edge → tìm water_y
           → dùng U2Net mask (realtime) làm bottle mask
           → tính fill_ratio → OK/NG theo ngưỡng
"""

import cv2
import numpy as np
import json
from pathlib import Path

TEMPLATE_FILENAME = "water_template.png"
CONFIG_FILENAME   = "liquid_config.json"


class LiquidLevelDetector:
    """
    Config (liquid_config.json):
        x          : tọa độ x góc trái ROI trên ảnh gốc
        w          : chiều rộng ROI
        h          : chiều cao template (mẫu mặt nước)
        baseline   : fill_ratio đo được trên ảnh mẫu (%)
        tolerance  : ± % cho phép lệch so với baseline
        min_fill   : baseline - tolerance  (tính tự động)
        max_fill   : baseline + tolerance  (tính tự động)
    """

    def __init__(self, project_root: Path):
        self.project_root  = Path(project_root)
        self.config_path   = self.project_root / CONFIG_FILENAME
        self.template_path = self.project_root / TEMPLATE_FILENAME

    # ----------------------------------------------------------------------- #
    #  SETUP                                                                   #
    # ----------------------------------------------------------------------- #
    def save_setup(self,
                   template_gray: np.ndarray,
                   x: int, w: int, h: int,
                   baseline: float,
                   tolerance: float = 10.0):
        """
        Lưu template + config.

        Args:
            template_gray : crop vùng mặt nước (grayscale) từ ảnh mẫu
            x, w, h       : tọa độ x, chiều rộng, chiều cao ROI trên ảnh gốc
            baseline      : fill_ratio đo được trên ảnh mẫu (%)
            tolerance     : ± % cho phép lệch (default 10%)
        """
        cv2.imwrite(str(self.template_path), template_gray)
        config = {
            "x"        : int(x),
            "w"        : int(w),
            "h"        : int(h),
            "baseline" : float(baseline),
            "tolerance": float(tolerance),
            "min_fill" : float(max(0.0,   baseline - tolerance)),
            "max_fill" : float(min(100.0, baseline + tolerance)),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(
            f"[LiquidLevel] 💾 Setup xong: "
            f"baseline={baseline:.1f}% ± {tolerance:.1f}% "
            f"→ range=[{config['min_fill']:.1f}, {config['max_fill']:.1f}]%"
        )

    def is_ready(self) -> bool:
        return self.config_path.exists() and self.template_path.exists()

    def load(self) -> tuple[dict, np.ndarray]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Không đọc được template: {self.template_path}")
        return config, template

    # ----------------------------------------------------------------------- #
    #  DETECT                                                                  #
    # ----------------------------------------------------------------------- #
    def detect(self, img_bgr: np.ndarray,
               object_mask: np.ndarray | None) -> dict | None:
        """
        Phát hiện mức chất lỏng.

        Args:
            img_bgr     : ảnh BGR từ camera (numpy HxWx3)
            object_mask : uint8 mask [0,255] từ U2Net, shape (H,W).
                          Nếu None → dùng toàn bộ cột ROI.

        Returns:
            dict:
                is_ok       : bool
                fill_ratio  : float (%)
                baseline    : float (%) — ngưỡng mẫu
                tolerance   : float (%)
                water_y     : int
                y_top       : int
                y_bottom    : int
                dist_water  : int (px)
                dist_void   : int (px)
                match_val   : float (độ khớp template)
                display_img : np.ndarray BGR (đã vẽ kết quả)
            hoặc None nếu không detect được.
        """
        if not self.is_ready():
            print("[LiquidLevel] ⚠ Chưa setup.")
            return None

        config, template = self.load()
        x_roi     = config["x"]
        w_roi     = config["w"]
        h_roi     = config["h"]
        min_fill  = config["min_fill"]
        max_fill  = config["max_fill"]
        baseline  = config["baseline"]
        tolerance = config["tolerance"]

        img_h, img_w = img_bgr.shape[:2]
        x_end = min(x_roi + w_roi, img_w)
        w_act = x_end - x_roi
        if w_act <= 0:
            print("[LiquidLevel] ⚠ ROI nằm ngoài ảnh.")
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ── Template matching trên Canny ──────────────────────────────────
        search_area = gray[:, x_roi:x_end]
        s_edges     = cv2.Canny(search_area, 50, 150)
        t_edges     = cv2.Canny(template,    50, 150)

        if t_edges.shape[1] > w_act:
            t_edges = cv2.resize(t_edges, (w_act, t_edges.shape[0]))

        if t_edges.shape[0] >= s_edges.shape[0] or t_edges.shape[1] > s_edges.shape[1]:
            print("[LiquidLevel] ⚠ Template lớn hơn vùng tìm kiếm.")
            return None

        res = cv2.matchTemplate(s_edges, t_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.1:
            print(f"[LiquidLevel] ⚠ Độ khớp thấp ({max_val:.3f}) — bỏ qua.")
            return None

        water_y = max_loc[1] + (h_roi // 2)

        # ── y_top / y_bottom từ U2Net mask ───────────────────────────────
        if object_mask is not None:
            mask_h, mask_w = object_mask.shape[:2]
            if (mask_h, mask_w) != (img_h, img_w):
                object_mask = cv2.resize(object_mask, (img_w, img_h),
                                         interpolation=cv2.INTER_NEAREST)
            col_mask  = object_mask[:, x_roi:x_end]
            bottle_ys = np.where(col_mask > 127)
            if len(bottle_ys[0]) > 0:
                y_top    = int(np.min(bottle_ys[0]))
                y_bottom = int(np.max(bottle_ys[0]))
            else:
                y_top, y_bottom = 0, img_h - 1
        else:
            y_top, y_bottom = 0, img_h - 1

        # ── Tính tỷ lệ ────────────────────────────────────────────────────
        total_h = y_bottom - y_top
        if total_h <= 0:
            print("[LiquidLevel] ⚠ Chiều cao chai = 0.")
            return None

        dist_void  = max(0, water_y - y_top)
        dist_water = max(0, y_bottom - water_y)
        fill_ratio = (dist_water / total_h) * 100.0
        is_ok      = (min_fill <= fill_ratio <= max_fill)
        is_ok = True

        # ── Vẽ kết quả ────────────────────────────────────────────────────
        display = self._draw(
            img_bgr, x_roi, w_act, water_y,
            y_top, y_bottom, dist_void, dist_water,
            fill_ratio, is_ok, baseline, tolerance, min_fill, max_fill
        )

        label = "✅ OK" if is_ok else "❌ NG"
        print(
            f"[LiquidLevel] {label}  "
            f"fill={fill_ratio:.1f}%  baseline={baseline:.1f}% ± {tolerance:.1f}%  "
            f"range=[{min_fill:.1f},{max_fill:.1f}]  match={max_val:.3f}"
        )

        return {
            "is_ok"      : is_ok,
            "fill_ratio" : fill_ratio,
            "baseline"   : baseline,
            "tolerance"  : tolerance,
            "min_fill"   : min_fill,
            "max_fill"   : max_fill,
            "water_y"    : water_y,
            "y_top"      : y_top,
            "y_bottom"   : y_bottom,
            "dist_water" : dist_water,
            "dist_void"  : dist_void,
            "match_val"  : float(max_val),
            "display_img": display,
        }

    # ----------------------------------------------------------------------- #
    #  DRAW — dùng chung cho cả setup preview lẫn inference                   #
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _draw(img_bgr,
              x_roi, w_roi, water_y,
              y_top, y_bottom,
              dist_void, dist_water,
              fill_ratio, is_ok,
              baseline, tolerance,
              min_fill, max_fill) -> np.ndarray:
        out   = img_bgr.copy()
        x_end = x_roi + w_roi
        x_mid = x_roi + w_roi // 2

        color_ok    = (0, 255, 0)
        color_ng    = (0, 0, 255)
        color_water = (255, 140, 0)   # cam đậm
        color_void  = (200, 200, 0)   # vàng
        color_base  = (255, 255, 255) # trắng — đường baseline

        res_color = color_ok if is_ok else color_ng

        # ── Đường mực nước (ngang) ────────────────────────────────────────
        cv2.line(out, (x_roi, water_y), (x_end, water_y), res_color, 2)

        # ── Đoạn trống (void, trên mực nước) ─────────────────────────────
        cv2.line(out, (x_mid, y_top),   (x_mid, water_y), color_void,  2)

        # ── Đoạn nước (dưới mực nước) ─────────────────────────────────────
        cv2.line(out, (x_mid, water_y), (x_mid, y_bottom), color_water, 2)

        # ── Đường baseline (đứt nét) ──────────────────────────────────────
        if y_top < y_bottom:
            baseline_y = int(y_bottom - (baseline / 100.0) * (y_bottom - y_top))
            # Vẽ đứt nét thủ công
            dash_len, gap_len = 8, 5
            xx = x_roi
            toggle = True
            while xx < x_end:
                end_xx = min(xx + (dash_len if toggle else gap_len), x_end)
                if toggle:
                    cv2.line(out, (xx, baseline_y), (end_xx, baseline_y), color_base, 1)
                xx     = end_xx
                toggle = not toggle

        # ── Text kết quả ──────────────────────────────────────────────────
        tx = x_end + 8

        # Nền mờ cho text
        text_lines = [
            (f"{'OK' if is_ok else 'NG'}  {fill_ratio:.1f}%", res_color, 0.65, 2),
            (f"Base: {baseline:.1f}% +/-{tolerance:.1f}%",    color_base, 0.45, 1),
            (f"Nuoc: {dist_water}px",                          color_water, 0.45, 1),
            (f"Trong: {dist_void}px",                          color_void,  0.45, 1),
        ]
        ty = water_y - 10
        for text, color, scale, thick in text_lines:
            cv2.putText(out, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            ty += int(scale * 38)

        return out

    # ----------------------------------------------------------------------- #
    #  DRAW OVERLAY — vẽ lên ảnh đã có (body overlay)                         #
    # ----------------------------------------------------------------------- #
    def draw_on_existing(self, existing_bgr: np.ndarray,
                         result: dict) -> np.ndarray:
        """
        Vẽ kết quả liquid lên ảnh đã có overlay (body inspection).
        Không copy ảnh gốc, vẽ thẳng lên existing_bgr.copy().

        Args:
            existing_bgr : ảnh đã có body overlay
            result       : dict trả về từ detect()
        """
        if result is None:
            return existing_bgr

        config, _ = self.load()
        x_roi  = config["x"]
        w_roi  = config["w"]

        return self._draw(
            existing_bgr,
            x_roi, w_roi,
            result["water_y"],
            result["y_top"],
            result["y_bottom"],
            result["dist_void"],
            result["dist_water"],
            result["fill_ratio"],
            result["is_ok"],
            result["baseline"],
            result["tolerance"],
            result["min_fill"],
            result["max_fill"],
        )