
# hardware/sorting/arduino_serial.py
import serial
import time
import threading
import os
import subprocess

class ArduinoSerial:
    def __init__(self, port="/dev/ttyACM0", baud=9600):
        self._lock = threading.Lock()
        
        # Tự kích hoạt module nếu chưa có
        self._ensure_module()
        
        # Tự tìm đúng cổng nếu port không tồn tại
        if not os.path.exists(port):
            port = self._find_arduino() or port
        
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print(f"[ArduinoSerial] ✅ Kết nối {port} @ {baud}")
        except serial.SerialException as e:
            self.ser = None
            print(f"[ArduinoSerial] ❌ Không kết nối được: {e}")

    def _ensure_module(self):
        """Load cdc_acm module nếu chưa load."""
        result = subprocess.run(["lsmod"], capture_output=True, text=True)
        if "cdc_acm" not in result.stdout:
            print("[ArduinoSerial] 🔧 Loading cdc_acm module...")
            subprocess.run(["sudo", "modprobe", "cdc_acm"], 
                         capture_output=True)
            time.sleep(1)

    def _find_arduino(self):
        """Tự tìm cổng Arduino theo vendor ID 2341."""
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if p.vid == 0x2341:
                print(f"[ArduinoSerial] 🔍 Tìm thấy Arduino: {p.device}")
                return p.device
        print("[ArduinoSerial] ⚠️  Không tìm thấy Arduino!")
        return None

    def send(self, value: int):
        """Gửi '0' hoặc '1' tới Arduino."""
        if self.ser is None:
            print(f"[ArduinoSerial] ⚠️  Chưa kết nối, bỏ qua lệnh {value}")
            return
        with self._lock:
            self.ser.write(str(value).encode())
            self.ser.flush()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[ArduinoSerial] 🔌 Đã đóng kết nối.")