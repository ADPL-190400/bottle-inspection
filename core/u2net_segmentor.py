"""
U2Net TensorRT Segmentor — tách vùng vật thể (chai).

Context management:
  - KHÔNG dùng pycuda.autoinit toàn cục
  - Tự tạo CUDAcontext, pop ngay sau init
  - Thread dùng U2Net phải gọi attach() trước / detach() sau
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

U2NET_SIZE     = 320
MASK_THRESH    = 0.5
FEATHER_RADIUS = 3
TRT_LOGGER     = trt.Logger(trt.Logger.WARNING)


class U2NetSegmentor:
    def __init__(self, engine_path: str):
        cuda.init()
        self._cuda_device  = cuda.Device(0)
        self._cuda_context = self._cuda_device.make_context()

        with open(engine_path, "rb") as f, \
             trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self._dev_ptrs   = {}
        self.input_name  = None
        self.output_name = None

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
                    self.output_name = name

        self.input_host,  self.input_dev,  _ = self._dev_ptrs[self.input_name]
        self.output_host, self.output_dev, _ = self._dev_ptrs[self.output_name]

        # Pop ngay — thread khác push lại khi cần
        self._cuda_context.pop()

        print(f"[U2Net] Loaded: {engine_path}  "
              f"in={self.engine.get_tensor_shape(self.input_name)}  "
              f"out={self.engine.get_tensor_shape(self.output_name)}")

    # ------------------------------------------------------------------ #
    def attach(self):
        """Gọi đầu thread trước khi inference."""
        self._cuda_context.push()

    def detach(self):
        """Gọi cuối thread sau khi inference xong."""
        self._cuda_context.pop()

    # ------------------------------------------------------------------ #
    def _infer_raw(self, img_bgr: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img_bgr.shape[:2]
        img = cv2.resize(img_bgr, (U2NET_SIZE, U2NET_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        img  = ((img - mean) / std).transpose(2, 0, 1)
        np.copyto(self.input_host, img.ravel())

        cuda.memcpy_htod_async(self.input_dev, self.input_host, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_dev, self.stream)
        self.stream.synchronize()

        pred = self.output_host.reshape(U2NET_SIZE, U2NET_SIZE)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return cv2.resize(pred, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # ------------------------------------------------------------------ #
    def get_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Trả về mask uint8 [0,255] cùng kích thước img_bgr.
        255 = vật thể, 0 = background.
        Yêu cầu attach() đã được gọi trong thread này.
        """
        mask_f = self._infer_raw(img_bgr)
        mask   = (mask_f >= MASK_THRESH).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

        if FEATHER_RADIUS > 0:
            r    = FEATHER_RADIUS * 2 + 1
            mask = cv2.GaussianBlur(mask, (r, r), 0)

        return mask

    def get_masks_batch(self, imgs_bgr: list) -> list:
        """
        Sequential cho nhiều ảnh. Yêu cầu attach() trước.
        """
        return [
            self.get_mask(img) if img is not None else None
            for img in imgs_bgr
        ]