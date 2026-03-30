import threading
import queue
import time
from hardware.gpio.trigger_input_sorting import TriggerSorting
from hardware.sorting.arduino_serial import ArduinoSerial  


class SortingActuator(threading.Thread):
    """
    Flow:
        PipelineManager → sorting_queue.put(batch_ok: bool)
                ↓
        [vật phẩm di chuyển đến vị trí sorting]
                ↓
        TriggerSorting (cảm biến vật lý) → trigger_queue.put(trigger_time)
                ↓
        SortingActuator.run()
            ├── OK → gửi '0' tới Arduino
            └── NG → gửi '1' tới Arduino  ← kích hoạt actuator
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        stop_event: threading.Event,
        arduino_port: str = "/dev/ttyACM0",
        arduino_baud: int = 9600,
    ):
        super().__init__(daemon=True, name="SortingActuator")
        self.input_queue = input_queue
        self.stop_event  = stop_event

        self._trigger_stop = threading.Event()
        self.trigger_queue = queue.Queue(maxsize=4)

        self.thread_trigger_sorting = TriggerSorting(
            self.trigger_queue,
            stop_event=self._trigger_stop,
            offset=0,
        )
        self.thread_trigger_sorting.start()

        # ── Arduino ──────────────────────────────────────────────────────── #
        self.arduino = ArduinoSerial(port=arduino_port, baud=arduino_baud)

    # ----------------------------------------------------------------------- #
    def run(self):
        print("[SortingActuator] ▶ Running... waiting for trigger")

        while not self.stop_event.is_set():

            # 1. Chờ trigger vật lý
            try:
                trigger_time = self.trigger_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 2. Lấy kết quả batch từ input_queue
            try:
                batch_ok = self.input_queue.get_nowait()
            except queue.Empty:
                print("[SortingActuator] ⚠️  Trigger nhưng không có kết quả AI trong queue!")
                continue

            # 3. Log
            label = "✅ OK" if batch_ok else "❌ NG"
            print(f"[SortingActuator] trigger={trigger_time} → {label}")

            # 4. Gửi lệnh tới Arduino
            #    OK → '0' (không làm gì)
            #    NG → '1' (kích hoạt actuator loại bỏ)
            self.arduino.send(0 if batch_ok else 1)

        print("[SortingActuator] 🛑 Kết thúc.")
        self._trigger_stop.set()
        self.arduino.close()     # ← đóng Serial khi dừng