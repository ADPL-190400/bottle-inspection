import serial, time

PORT = "/dev/ttyACM0"
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print(f"Đã kết nối: {ser.name}")

try:
    while True:
        lenh = input("Nhập 0 hoặc 1 (q thoát): ").strip()
        if lenh == 'q':
            break
        if lenh in ['0', '1']:
            ser.write(lenh.encode())
            print(f"Đã gửi: {lenh}")
        else:
            print("Chỉ nhập 0 hoặc 1")
finally:
    ser.close()
    print("Đã đóng kết nối.")