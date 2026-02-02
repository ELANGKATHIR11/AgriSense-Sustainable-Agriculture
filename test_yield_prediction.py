import sys
import json
import subprocess

# Test data for predict_yield.py
# ['Area_Ha', 'label_encoded', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall']
# Input format expected by predict_yield.py:
# {'crop': 'Rice', 'area': 1.5, 'N': 100, 'P': 50, 'K': 50, 'temperature': 30, 'humidity': 80, 'rainfall': 200, 'area_unit': 'Acres'}

input_data = {
    "crop": "Rice",
    "area": 2.0,
    "area_unit": "Hectare",
    "N": 80,
    "P": 40,
    "K": 40,
    "temperature": 28,
    "humidity": 70,
    "rainfall": 150,
}

input_json = json.dumps(input_data)

try:
    print("Testing predict_yield.py...")
    # Run the script via subprocess, passing input to stdin
    process = subprocess.Popen(
        ["python", "backend/ml/predict_yield.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = process.communicate(input=input_json)

    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    print("Exit Code:", process.returncode)

except Exception as e:
    print(f"Error: {e}")
