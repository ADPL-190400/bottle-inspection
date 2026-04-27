import os
import time
import json
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
except (ImportError, ValueError):
    Aravis = None


class CameraError(RuntimeError):
    pass


@dataclass
class CameraDescriptor:
    device_id: str
    address: str = ""
    model_name: str = ""
    vendor_name: str = ""

    @property
    def display_name(self) -> str:
        if self.model_name:
            return f"{self.model_name} [{self.device_id}]"
        return self.device_id


class GenericCamera:
    BAYER_FORMAT_MAP = {
        0x0108000A: cv2.COLOR_BayerRG2BGR,
        0x0108000B: cv2.COLOR_BayerGB2BGR,
        0x0108000C: cv2.COLOR_BayerGR2BGR,
        0x01080009: cv2.COLOR_BayerBG2BGR,
    }

    def __init__(self, descriptor: CameraDescriptor, buffer_count: int = 5):
        self._ensure_backend()
        self.descriptor = descriptor
        self._camera = Aravis.Camera.new(descriptor.device_id)
        if self._camera is None:
            raise CameraError(f"Unable to open camera: {descriptor.device_id}")
        self._device = self._camera.get_device()
        self._stream = None
        self._buffer_count = buffer_count
        self._pixel_format = self.get_pixel_format()
        self._flip_x = False
        self._flip_y = False

    @staticmethod
    def _ensure_backend():
        if Aravis is None:
            raise CameraError(
                "Aravis backend is not available. Install gir1.2-aravis-0.8 and python3-gi."
            )

    @classmethod
    def enumerate_devices(cls) -> list[CameraDescriptor]:
        cls._ensure_backend()
        Aravis.update_device_list()
        devices = []
        for index in range(Aravis.get_n_devices()):
            device_id = Aravis.get_device_id(index)
            address = ""
            try:
                address = Aravis.get_device_address(index)
            except Exception:
                pass
            devices.append(
                CameraDescriptor(
                    device_id=device_id,
                    address=address or "",
                )
            )
        return devices

    def close(self):
        try:
            self.stop_acquisition()
        finally:
            self._stream = None
            self._camera = None
            self._device = None

    def get_model_name(self) -> str:
        try:
            return self._camera.get_model_name() or ""
        except Exception:
            return ""

    def get_vendor_name(self) -> str:
        try:
            return self._camera.get_vendor_name() or ""
        except Exception:
            return ""

    def get_pixel_format(self) -> int:
        try:
            return int(self._camera.get_pixel_format())
        except Exception:
            return 0x01080001

    def get_pixel_format_as_string(self) -> str:
        try:
            return self._camera.get_pixel_format_as_string()
        except Exception:
            return "Unknown"

    def _get_feature_range(self, name: str) -> dict:
        try:
            node = self._device.get_feature(name)
            current = self._device.get_integer_feature_value(name)
            return {
                "current": int(current),
                "min": int(node.get_min()),
                "max": int(node.get_max()),
                "inc": int(node.get_inc() or 1),
            }
        except Exception:
            return {"current": 0, "min": 0, "max": 0, "inc": 1}

    def get_camera_info(self) -> dict:
        info = {
            "Width": self._get_feature_range("Width"),
            "Height": self._get_feature_range("Height"),
            "OffsetX": self._get_feature_range("OffsetX"),
            "OffsetY": self._get_feature_range("OffsetY"),
            "pixel_format": self.get_pixel_format(),
            "pixel_format_str": self.get_pixel_format_as_string(),
            "model_name": self.get_model_name(),
            "vendor_name": self.get_vendor_name(),
        }
        info["trigger_nodes"] = {}
        for node_name in ("TriggerSelector", "TriggerMode", "TriggerSource", "TriggerSoftware"):
            try:
                info["trigger_nodes"][node_name] = self._device.get_feature(node_name) is not None
            except Exception:
                info["trigger_nodes"][node_name] = False
        return info

    @staticmethod
    def _align(value: int, inc: int) -> int:
        if inc <= 1:
            return int(value)
        return int(value // inc * inc)

    @staticmethod
    def config_key(device_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in device_id)

    def set_exposure_time(self, exposure_us: float):
        try:
            self._camera.set_exposure_time_auto(Aravis.Auto.OFF)
        except Exception:
            pass
        exp_min, exp_max = self._camera.get_exposure_time_bounds()
        exposure_us = max(float(exp_min), min(float(exp_max), float(exposure_us)))
        self._camera.set_exposure_time(exposure_us)

    def get_exposure_time(self) -> float:
        return float(self._camera.get_exposure_time())

    def set_roi(self, offset_x: int, offset_y: int, width: int, height: int, cam_info: dict | None = None):
        if cam_info is None:
            cam_info = self.get_camera_info()
        restart_acquisition = self._stream is not None

        max_w = cam_info["Width"]["max"]
        max_h = cam_info["Height"]["max"]
        inc_w = cam_info["Width"]["inc"] or 1
        inc_h = cam_info["Height"]["inc"] or 1
        inc_ox = cam_info["OffsetX"]["inc"] or 1
        inc_oy = cam_info["OffsetY"]["inc"] or 1

        width = self._align(width, inc_w)
        height = self._align(height, inc_h)
        offset_x = self._align(offset_x, inc_ox)
        offset_y = self._align(offset_y, inc_oy)

        width = max(inc_w, min(width, max_w))
        height = max(inc_h, min(height, max_h))
        offset_x = max(0, min(offset_x, max_w - width))
        offset_y = max(0, min(offset_y, max_h - height))

        if restart_acquisition:
            self.stop_acquisition()
            self._stream = None

        try:
            self._device.set_integer_feature_value("OffsetX", 0)
            self._device.set_integer_feature_value("OffsetY", 0)
            self._device.set_integer_feature_value("Width", width)
            self._device.set_integer_feature_value("Height", height)
            self._device.set_integer_feature_value("OffsetX", offset_x)
            self._device.set_integer_feature_value("OffsetY", offset_y)
        finally:
            if restart_acquisition:
                self.start_acquisition()

    def get_region(self) -> tuple[int, int, int, int]:
        region = self._camera.get_region()
        return tuple(int(v) for v in region)

    def enable_software_trigger(self, cam_info: dict | None = None):
        if cam_info is None:
            cam_info = self.get_camera_info()
        if cam_info["trigger_nodes"].get("TriggerSelector"):
            try:
                self._device.set_string_feature_value("TriggerSelector", "FrameStart")
            except Exception:
                pass
        self._device.set_string_feature_value("TriggerMode", "On")
        self._device.set_string_feature_value("TriggerSource", "Software")
        time.sleep(0.1)

    def execute_software_trigger(self):
        self._device.execute_command("TriggerSoftware")

    def start_acquisition(self):
        if self._stream is not None:
            return
        self._stream = self._camera.create_stream(None, None)
        if self._stream is None:
            raise CameraError(f"Unable to create stream for {self.descriptor.device_id}")
        payload = int(self._camera.get_payload())
        for _ in range(self._buffer_count):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self._camera.start_acquisition()

    def stop_acquisition(self):
        if self._camera is None:
            return
        try:
            self._camera.stop_acquisition()
        except Exception:
            pass

    def _decode_frame_bgr(self, data: bytes, width: int, height: int) -> np.ndarray | None:
        pixel_format = self._pixel_format
        raw = np.frombuffer(data, dtype=np.uint8)

        if pixel_format in (0x010C0020, 0x010C001E):
            img = raw.reshape(height, width, 3).copy()
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if pixel_format == 0x010C0021:
            return raw.reshape(height, width, 3).copy()

        if pixel_format == 0x01080001:
            mono = raw.reshape(height, width)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        bayer_code = self.BAYER_FORMAT_MAP.get(pixel_format)
        if bayer_code is not None:
            return cv2.cvtColor(raw.reshape(height, width).copy(), bayer_code)

        try:
            return cv2.cvtColor(raw.reshape(height, width).copy(), cv2.COLOR_BayerGB2BGR)
        except Exception:
            return None

    def grab_frame(self, timeout_us: int = 3_000_000, color_order: str = "bgr") -> np.ndarray | None:
        if self._stream is None:
            self.start_acquisition()

        buf = self._stream.timeout_pop_buffer(timeout_us)
        if buf is None:
            return None

        x, y, width, height = self.get_region()
        _ = (x, y)
        try:
            try:
                data = buf.get_part_data(0)
            except Exception:
                data = buf.get_data()
            frame_bgr = self._decode_frame_bgr(data, width, height)
        finally:
            self._stream.push_buffer(buf)

        if frame_bgr is None:
            return None
        if self._flip_x:
            frame_bgr = cv2.flip(frame_bgr, 0)
        if self._flip_y:
            frame_bgr = cv2.flip(frame_bgr, 1)
        if color_order.lower() == "rgb":
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_bgr

    def export_config(self) -> dict:
        x, y, width, height = self.get_region()
        return {
            "device_id": self.descriptor.device_id,
            "exposure_time": self.get_exposure_time(),
            "roi": {
                "offset_x": x,
                "offset_y": y,
                "width": width,
                "height": height,
            },
            "pixel_format": self.get_pixel_format_as_string(),
            "flip_x": self._flip_x,
            "flip_y": self._flip_y,
        }

    def apply_config(self, config: dict):
        exposure_time = config.get("exposure_time")
        if exposure_time is not None:
            self.set_exposure_time(exposure_time)

        roi = config.get("roi") or {}
        if roi:
            self.set_roi(
                int(roi.get("offset_x", 0)),
                int(roi.get("offset_y", 0)),
                int(roi.get("width", self.get_camera_info()["Width"]["current"])),
                int(roi.get("height", self.get_camera_info()["Height"]["current"])),
            )

        self.set_flip(
            bool(config.get("flip_x", False)),
            bool(config.get("flip_y", False)),
        )
        self._pixel_format = self.get_pixel_format()

    def set_flip(self, flip_x: bool = False, flip_y: bool = False):
        self._flip_x = bool(flip_x)
        self._flip_y = bool(flip_y)

    def get_flip(self) -> tuple[bool, bool]:
        return self._flip_x, self._flip_y


def load_camera_config(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_camera_config(path: str, config: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
