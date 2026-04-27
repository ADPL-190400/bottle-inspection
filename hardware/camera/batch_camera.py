import os
import queue
import threading
import time

from core.path_manager import BASE_DIR
from hardware.camera.generic_camera import GenericCamera, load_camera_config
from hardware.gpio.trigger_input_camera import TriggerCamera


class BatchCamera(threading.Thread):
    def __init__(self, output_queue, stop_event, infor_project):
        super().__init__()
        self.cameras = []
        self.camera_positions = []
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.infor_project = infor_project or {}
        self.project_name = self.infor_project.get("project_name")

        self.init_cameras()
        self.init_trigger_camera()

    def init_cameras(self):
        print("Init Cameras")
        dev_list = GenericCamera.enumerate_devices()
        print(f"[BatchCamera] Detected {len(dev_list)} camera(s)")

        if not dev_list:
            print("No camera found")
            return

        for dev_info in dev_list:
            try:
                camera = GenericCamera(dev_info)
                config_path = os.path.join(
                    BASE_DIR,
                    "projects",
                    self.project_name or "",
                    "camera_config",
                    f"{GenericCamera.config_key(dev_info.device_id)}.json",
                )
                config = load_camera_config(config_path)
                if config is not None:
                    print("config file camera", config_path)
                    camera.apply_config(config)
                position = str((config or {}).get("camera_position", f"cam{len(self.cameras) + 1}"))
                camera.enable_software_trigger()
                camera.start_acquisition()
                self.cameras.append(camera)
                self.camera_positions.append(position)
            except Exception as e:
                print("CameraInit Failed:", e)

    def init_trigger_camera(self):
        stop_event = threading.Event()
        self.trigger_queue = queue.Queue(maxsize=1)
        self.thread_trigger_camera = TriggerCamera(
            self.trigger_queue, stop_event=stop_event, offset=3
        )
        self.thread_trigger_camera.start()

    def capture_all_sync(self):
        frames = []

        for cam in self.cameras:
            try:
                cam.execute_software_trigger()
            except Exception as e:
                print("Trigger error:", e)

        for cam in self.cameras:
            try:
                frames.append(cam.grab_frame(timeout_us=200_000, color_order="rgb"))
            except Exception as e:
                print("Capture error:", e)
                frames.append(None)

        return frames

    def run(self):
        print("[BatchCamera] Running... waiting for trigger")

        while not self.stop_event.is_set():
            try:
                if self.trigger_queue.empty():
                    time.sleep(0.01)
                    continue

                trigger_time = self.trigger_queue.get()
                frames = self.capture_all_sync()

                try:
                    self.output_queue.put_nowait((trigger_time, frames))
                except queue.Full:
                    print("[BatchCamera] Output queue full, dropping")

            except Exception as e:
                print(f"[BatchCamera Error] {e}")

        for cam in self.cameras:
            cam.close()

        self.thread_trigger_camera.stop_event.set()

    def get_camera_positions(self):
        return list(self.camera_positions)
