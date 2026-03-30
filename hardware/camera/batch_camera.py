<<<<<<< HEAD
# #coding=utf-8
# import threading
# from core.path_manager import BASE_DIR
# import os
# import platform
# import cv2
# import numpy as np
# from hardware.camera import mvsdk
# import queue
# import time
# import keyboard

# class BatchCamera(threading.Thread):
#     def __init__(self, output_queue,stop_event):
#         super().__init__()
#         self.cameras = []
#         self.buffers = []
#         self.mono_flags = []
#         self.output_queue = output_queue
#         self.stop_event = stop_event

#         # self.trigger_event = threading.Event()

#         self.init_cameras()

#     def init_cameras(self):
#         DevList = mvsdk.CameraEnumerateDevice()
#         if len(DevList) < 1:
#             print("No camera found")
#             return

#         for i, DevInfo in enumerate(DevList):
#             # print(f"{i}: {DevInfo.acFriendlyName} {DevInfo.acDisplayName}")

#             try:
#                 hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
#             except mvsdk.CameraException as e:
#                 print("CameraInit Failed:", e.message)
#                 continue

#             # load config (có thể thay theo từng cam)
#             config_path = os.path.join(BASE_DIR, "3.config")
#             if os.path.exists(config_path):
#                 mvsdk.CameraReadParameterFromFile(hCamera, config_path)

#             cap = mvsdk.CameraGetCapability(hCamera)
#             mono = (cap.sIspCapacity.bMonoSensor != 0)

#             # # format
#             # if mono:
#             #     mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
#             # else:
#             #     mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

#             # # trigger mode
#             # mvsdk.CameraSetTriggerMode(hCamera, 1)

#             # # exposure
#             # mvsdk.CameraSetAeState(hCamera, 0)
#             # mvsdk.CameraSetExposureTime(hCamera, 5000)

#             mvsdk.CameraPlay(hCamera)

#             # buffer
#             FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if mono else 3)
#             pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

#             # save
#             self.cameras.append(hCamera)
#             self.buffers.append(pFrameBuffer)
#             self.mono_flags.append(mono)



#     def capture_one(self, hCamera, pFrameBuffer, mono):
#         try:
#             mvsdk.CameraSoftTrigger(hCamera)

#             pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)

#             mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
#             mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

#             if platform.system() == "Windows":
#                 mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

#             frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
#             frame = np.frombuffer(frame_data, dtype=np.uint8)

#             frame = frame.reshape(
#                 (FrameHead.iHeight, FrameHead.iWidth, 1 if mono else 3)
#             )

#             # frame = cv2.resize(frame, (640, 480))

#             return frame

#         except mvsdk.CameraException as e:
#             print("Capture error:", e.message)
#             return None


#     # def trigger(self):
#     #     self.trigger_event.set()


#     def run(self):
#         print("SPACE = trigger all cameras")
       

#         while self.stop_event.is_set() == False:
#             # if not self.trigger_event.wait(timeout=0.1):
#             #     continue

#             # # reset trigger
#             # self.trigger_event.clear()
#             if keyboard.is_pressed('space'):
#                 print("Triggered! Capturing...")

#                 frames = []
#                 for i, (cam, buf, mono) in enumerate(zip(self.cameras, self.buffers, self.mono_flags)):
#                     frame = self.capture_one(cam, buf, mono)

#                     if frame is not None:
#                         frames.append(frame)
#                     else:
#                         frames.append(None)

#                 try:
#                     self.output_queue.put_nowait(('trigger_id', frames))
#                 except queue.Full:
#                     try:
#                         self.output_queue.get_nowait()
#                         self.output_queue.put_nowait(('trigger_id', frames))
#                     except queue.Empty:
#                         pass
                
#             time.sleep(0.01)


#         for i, (cam, buf, mono) in enumerate(zip(self.cameras, self.buffers, self.mono_flags)):
#             mvsdk.CameraAlignFree(buf)
#             mvsdk.CameraUnInit(cam)
            






#coding=utf-8
=======
>>>>>>> 5fe2763 (update 2703)
import threading
from core.path_manager import BASE_DIR
import os
import platform
import numpy as np
from hardware.camera import mvsdk
from hardware.gpio.trigger_input_camera import TriggerCamera
import queue
import time
<<<<<<< HEAD

=======
from PIL import Image
>>>>>>> 5fe2763 (update 2703)


class BatchCamera(threading.Thread):

    def __init__(self, output_queue, stop_event):
        super().__init__()

<<<<<<< HEAD
        self.cameras = []
        self.buffers = []
        self.mono_flags = []

        self.output_queue = output_queue
        self.stop_event = stop_event

        
        # self.trigger_queue = queue.Queue(maxsize=1)
=======
        self.cameras    = []
        self.buffers    = []
        self.mono_flags = []

        self.output_queue = output_queue
        self.stop_event   = stop_event
>>>>>>> 5fe2763 (update 2703)

        self.init_cameras()
        self.init_trigger_camera()

    # ================= INIT =================

    def init_cameras(self):
        print('Init Cameras')
        DevList = mvsdk.CameraEnumerateDevice()
        print(f"[BatchCamera] Detected {len(DevList)} camera(s)")

<<<<<<< HEAD


=======
>>>>>>> 5fe2763 (update 2703)
        if len(DevList) < 1:
            print("No camera found")
            return

        for DevInfo in DevList:
            try:
                hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            except mvsdk.CameraException as e:
                print("CameraInit Failed:", e.message)
                continue

<<<<<<< HEAD
            config_path = os.path.join(BASE_DIR, "3.config")
            if os.path.exists(config_path):
                mvsdk.CameraReadParameterFromFile(hCamera, config_path)

            cap = mvsdk.CameraGetCapability(hCamera)
            mono = (cap.sIspCapacity.bMonoSensor != 0)

            # bật software trigger mode
            mvsdk.CameraSetTriggerMode(hCamera, 1)

            mvsdk.CameraPlay(hCamera)

            FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if mono else 3)
=======
            config_path = os.path.join(BASE_DIR, "19.config")
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
>>>>>>> 5fe2763 (update 2703)
            pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

            self.cameras.append(hCamera)
            self.buffers.append(pFrameBuffer)
            self.mono_flags.append(mono)

<<<<<<< HEAD

    def init_trigger_camera(self):
        stop_event = threading.Event()
        self.trigger_queue = queue.Queue(maxsize=1)
        self.thread_trigger_camera = TriggerCamera( self.trigger_queue,stop_event=stop_event, offset=3)
        self.thread_trigger_camera.start()



=======
    def init_trigger_camera(self):
        stop_event         = threading.Event()
        self.trigger_queue = queue.Queue(maxsize=1)
        self.thread_trigger_camera = TriggerCamera(
            self.trigger_queue, stop_event=stop_event, offset=3
        )
        self.thread_trigger_camera.start()

>>>>>>> 5fe2763 (update 2703)
    # ================= CORE =================

    def capture_all_sync(self):
        """
<<<<<<< HEAD
        Trigger tất cả camera → lấy frame
        """

        frames = []

        # 1. trigger tất cả camera gần như cùng lúc
        for cam in self.cameras:
            mvsdk.CameraSoftTrigger(cam)

        # 2. lấy frame ngay sau đó
=======
        Trigger tất cả camera → lấy frame.
        ✅ FIX: dùng .copy() để array tự own memory,
                tránh bị ghi đè khi camera capture frame tiếp theo.
        """
        frames = []

        # 1. Trigger tất cả camera gần như cùng lúc
        for cam in self.cameras:
            mvsdk.CameraSoftTrigger(cam)

        # 2. Lấy frame ngay sau đó
>>>>>>> 5fe2763 (update 2703)
        for cam, buf, mono in zip(self.cameras, self.buffers, self.mono_flags):
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(cam, 200)

                mvsdk.CameraImageProcess(cam, pRawData, buf, FrameHead)
                mvsdk.CameraReleaseImageBuffer(cam, pRawData)

                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(buf, FrameHead, 1)

                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(buf)
<<<<<<< HEAD
                frame = np.frombuffer(frame_data, dtype=np.uint8)
=======

                # ✅ FIX CHÍNH: .copy() để array own memory độc lập
                # Không có .copy() → array trỏ thẳng vào buf (C pointer)
                # → bị ghi đè khi trigger tiếp theo → data hỏng → score sai
                frame = np.frombuffer(frame_data, dtype=np.uint8).copy()
>>>>>>> 5fe2763 (update 2703)

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
<<<<<<< HEAD

=======
>>>>>>> 5fe2763 (update 2703)
        print("[BatchCamera] Running... waiting for trigger")

        while not self.stop_event.is_set():
            try:
                if self.trigger_queue.empty():
                    time.sleep(0.01)
                    continue

                trigger_time = self.trigger_queue.get()
<<<<<<< HEAD

                frames = self.capture_all_sync()

                # push sang pipeline
=======
                frames       = self.capture_all_sync()

>>>>>>> 5fe2763 (update 2703)
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

<<<<<<< HEAD
        # stop thread trigger camera
        self.thread_trigger_camera.stop_event.set()
        
=======
        self.thread_trigger_camera.stop_event.set()
>>>>>>> 5fe2763 (update 2703)
