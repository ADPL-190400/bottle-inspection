# hardware/arduino/arduino_serial.py
import serial
import time
import threading

class ArduinoSerial:
    def __init__(self, port="/dev/ttyACM0", baud=9600):
        self._lock = threading.Lock()
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Chờ Arduino reset
            print(f"[ArduinoSerial] ✅ Kết nối {port} @ {baud}")
        except serial.SerialException as e:
            self.ser = None
            print(f"[ArduinoSerial] ❌ Không kết nối được: {e}")

    def send(self, value: int):
        """Gửi '0' hoặc '1' tới Arduino."""
        if self.ser is None:
            print(f"[ArduinoSerial] ⚠️  Serial chưa kết nối, bỏ qua lệnh {value}")
            return
        with self._lock:
            self.ser.write(str(value).encode())
            self.ser.flush()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[ArduinoSerial] 🔌 Đã đóng kết nối.")