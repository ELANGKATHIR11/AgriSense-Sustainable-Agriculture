# Arduino Nano Firmware Stub

This folder contains a small Python bridge stub that reads serial output
from an Arduino Nano and forwards sensor readings to the AgriSense backend.

Files:
- `arduino_bridge.py` — example Python bridge using `pyserial` and `requests`.

Usage:
- Install dependencies: `pip install pyserial requests`
- Adjust `SERIAL_PORT` and `BACKEND_URL` in `arduino_bridge.py` and run:

```powershell
python AGRISENSE_IoT\\arduino_nano_firmware\\arduino_bridge.py
```

Replace with compiled Arduino firmware and a proper serial protocol in production.
