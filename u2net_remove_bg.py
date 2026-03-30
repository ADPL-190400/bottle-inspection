
"""
U2Net TensorRT pipeline cho Jetson Orin
Yêu cầu: tensorrt, pycuda
  pip install pycuda
  (tensorrt đã có sẵn trên JetPack)
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import sys

U2NET_SIZE     = 320
MASK_THRESH    = 0.5
FEATHER_RADIUS = 3
TRT_ENGINE_PATH = "models/u2netp.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class U2NetTRT:
    def __init__(self, engine_path: str):
        # Load engine
        with open(engine_path, "rb") as f, \
             trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        self.input_name  = None
        self.output_name = None
        self._dev_ptrs   = {}

        # Allocate buffers cho TẤT CẢ tensors (input + output)
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode  = self.engine.get_tensor_mode(name)
            size  = int(np.prod(shape))

            host = cuda.pagelocked_empty(size, np.float32)
            dev  = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            self._dev_ptrs[name] = (host, dev, shape)

            if mode == trt.TensorIOMode.INPUT:
                if self.input_name is None:
                    self.input_name = name
            else:
                if self.output_name is None:
                    self.output_name = name  # output[0] = mask chính

        self.input_host,  self.input_dev,  _ = self._dev_ptrs[self.input_name]
        self.output_host, self.output_dev, _ = self._dev_ptrs[self.output_name]

        print(f"[TRT] Engine loaded : {engine_path}")
        print(f"[TRT] Input  tensor : {self.input_name}  "
              f"{self.engine.get_tensor_shape(self.input_name)}")
        print(f"[TRT] Output tensor : {self.output_name} "
              f"{self.engine.get_tensor_shape(self.output_name)}")
        print(f"[TRT] Total tensors : {self.engine.num_io_tensors}")

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        """Trả về mask float32 [0,1] kích thước gốc."""
        orig_h, orig_w = img_bgr.shape[:2]

        # ── Preprocess ──
        img = cv2.resize(img_bgr, (U2NET_SIZE, U2NET_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        img  = ((img - mean) / std).transpose(2, 0, 1)  # HWC → CHW
        np.copyto(self.input_host, img.ravel())

        # ── H2D → Inference → D2H ──
        cuda.memcpy_htod_async(self.input_dev, self.input_host, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_dev, self.stream)
        self.stream.synchronize()

        # ── Postprocess ──
        pred = self.output_host.reshape(U2NET_SIZE, U2NET_SIZE)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        mask = cv2.resize(pred, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR)
        return mask.astype(np.float32)

    def build_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Trả về mask uint8 [0,255] đã xử lý hậu kỳ."""
        mask_f = self.infer(img_bgr)

        # Threshold
        mask = (mask_f >= MASK_THRESH).astype(np.uint8) * 255

        # Morphology — làm sạch viền
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

        # Feather — làm mềm viền
        if FEATHER_RADIUS > 0:
            r = FEATHER_RADIUS * 2 + 1
            mask = cv2.GaussianBlur(mask, (r, r), 0)

        return mask


def composite(bg: np.ndarray, obj: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
    """Alpha composite obj lên bg theo mask."""
    if bg.shape != obj.shape:
        obj = cv2.resize(obj, (bg.shape[1], bg.shape[0]))
    alpha  = mask.astype(np.float32) / 255.0
    alpha3 = np.stack([alpha] * 3, axis=-1)
    out    = obj * alpha3 + bg * (1 - alpha3)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    bg_path  = sys.argv[1] if len(sys.argv) > 1 else "models/bg2.jpg"
    obj_path = sys.argv[2] if len(sys.argv) > 2 else "models/2.jpg"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "result.png"

    # Kiểm tra file tồn tại
    import os
    for p in [bg_path, obj_path]:
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    model = U2NetTRT(TRT_ENGINE_PATH)

    bg    = cv2.imread(bg_path)
    frame = cv2.imread(obj_path)

    if bg is None or frame is None:
        print("[ERROR] Cannot read image files")
        sys.exit(1)

    # Warmup
    print("[WARMUP] Running 3 warmup iterations ...")
    for _ in range(3):
        model.infer(frame)
    print("[WARMUP] Done")

    # Benchmark
    print("[BENCH] Running 20 iterations ...")
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        mask = model.build_mask(frame)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"[PERF] avg={np.mean(times):.1f}ms  "
          f"min={np.min(times):.1f}ms  "
          f"max={np.max(times):.1f}ms  "
          f"fps={1000/np.mean(times):.1f}")

    # Lưu kết quả
    result = composite(bg, frame, mask)
    cv2.imwrite("debug_mask.png", mask)
    cv2.imwrite(out_path, result)
    print(f"[OK] Mask  saved : debug_mask.png")
    print(f"[OK] Result saved: {out_path}")


if __name__ == "__main__":
    main()