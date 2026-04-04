import threading
from core.path_manager import BASE_DIR
import os
import platform
import numpy as np
from hardware.camera import mvsdk
from hardware.gpio.trigger_input_camera import TriggerCamera
import queue
import time



class BatchCamera(threading.Thread):

    def __init__(self, output_queue, stop_event, infor_project):
        super().__init__()

        self.cameras    = []
        self.buffers    = []
        self.mono_flags = []

        self.output_queue = output_queue
        self.stop_event   = stop_event
        self.infor_project = infor_project
        self.project_name = self.infor_project.get("project_name", None)

        self.init_cameras()
        self.init_trigger_camera()

    # ================= INIT =================

    def init_cameras(self):
        print('Init Cameras')
        DevList = mvsdk.CameraEnumerateDevice()
        print(f"[BatchCamera] Detected {len(DevList)} camera(s)")

        if len(DevList) < 1:
            print("No camera found")
            return
        
        for DevInfo in DevList:
            try:
                hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            except mvsdk.CameraException as e:
                print("CameraInit Failed:", e.message)
                continue
            link_name = DevInfo.acSn.decode() if isinstance(DevInfo.acSn, bytes) else DevInfo.acSn
            print('link name',link_name)
            file_conf = f"{link_name}.config"
            config_path = os.path.join(BASE_DIR,"projects",self.project_name, "camera_config", file_conf)
            print('config camera path',config_path)
            if os.path.exists(config_path):
                print('config file camera', config_path)
                mvsdk.CameraReadParameterFromFile(hCamera, config_path)     

            cap  = mvsdk.CameraGetCapability(hCamera)
            mono = (cap.sIspCapacity.bMonoSensor != 0)

            mvsdk.CameraSetTriggerMode(hCamera, 1)
            mvsdk.CameraPlay(hCamera)

            FrameBufferSize = (
                cap.sResolutionRange.iWidthMax
                * cap.sResolutionRange.iHeightMax
                * (1 if mono else 3)
            )
            pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

            self.cameras.append(hCamera)
            self.buffers.append(pFrameBuffer)
            self.mono_flags.append(mono)

    def init_trigger_camera(self):
        stop_event         = threading.Event()
        self.trigger_queue = queue.Queue(maxsize=1)
        self.thread_trigger_camera = TriggerCamera(
            self.trigger_queue, stop_event=stop_event, offset=3
        )
        self.thread_trigger_camera.start()

    # ================= CORE =================

    def capture_all_sync(self):
        """
        Trigger tất cả camera → lấy frame.
        ✅ FIX: dùng .copy() để array tự own memory,
                tránh bị ghi đè khi camera capture frame tiếp theo.
        """
        frames = []

        # 1. Trigger tất cả camera gần như cùng lúc
        for cam in self.cameras:
            mvsdk.CameraSoftTrigger(cam)

        # 2. Lấy frame ngay sau đó
        for cam, buf, mono in zip(self.cameras, self.buffers, self.mono_flags):
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(cam, 200)

                mvsdk.CameraImageProcess(cam, pRawData, buf, FrameHead)
                mvsdk.CameraReleaseImageBuffer(cam, pRawData)

                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(buf, FrameHead, 1)

                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(buf)

                # ✅ FIX CHÍNH: .copy() để array own memory độc lập
                # Không có .copy() → array trỏ thẳng vào buf (C pointer)
                # → bị ghi đè khi trigger tiếp theo → data hỏng → score sai
                frame = np.frombuffer(frame_data, dtype=np.uint8).copy()

                frame = frame.reshape(
                    (FrameHead.iHeight, FrameHead.iWidth, 1 if mono else 3)
                )

                frames.append(frame)

            except mvsdk.CameraException as e:
                print("Capture error:", e.message)
                frames.append(None)

        return frames

    # ================= THREAD =================

    def run(self):
        print("[BatchCamera] Running... waiting for trigger")

        while not self.stop_event.is_set():
            try:
                if self.trigger_queue.empty():
                    time.sleep(0.01)
                    continue

                trigger_time = self.trigger_queue.get()
                frames       = self.capture_all_sync()

                try:
                    self.output_queue.put_nowait((trigger_time, frames))
                except queue.Full:
                    print("[BatchCamera] Output queue full, dropping")

            except Exception as e:
                print(f"[BatchCamera Error] {e}")

        # cleanup
        for cam, buf in zip(self.cameras, self.buffers):
            mvsdk.CameraAlignFree(buf)
            mvsdk.CameraUnInit(cam)

        self.thread_trigger_camera.stop_event.set()