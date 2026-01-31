"""
Arduino Nano bridge stub
- Reads serial data from Arduino and forwards to the backend HTTP endpoint.
- Replace with real serial parsing and error handling as needed.
"""

import time
import requests
import serial

SERIAL_PORT = "COM3"  # adjust for platform
BAUDRATE = 9600
BACKEND_URL = "http://localhost:8000/api/sensors"


def main():
    try:
        with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1) as ser:
            print(f"Listening on {SERIAL_PORT} @ {BAUDRATE}")
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    time.sleep(0.1)
                    continue
                print("Received:", line)
                try:
                    payload = {"device_id": "arduino-nano-stub", "data": line}
                    requests.post(BACKEND_URL, json=payload, timeout=5)
                except Exception as e:
                    print("Failed to forward to backend:", e)
    except serial.SerialException as e:
        print("Serial port error (stub):", e)


if __name__ == '__main__':
    main()
