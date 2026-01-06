
import subprocess
import sys
import time
import os

def main():
    print("🚀 Launching AgriSense Backend...")
    
    backend_dir = os.path.join("src", "backend")
    script_path = "start_fixed.py"
    
    # Run in a separate process
    process = subprocess.Popen(
        [sys.executable, script_path],
        cwd=backend_dir,
        shell=True
    )
    
    print(f"Backend process started with PID {process.pid}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print(f"Backend process exited with code {process.returncode}")
                break
    except KeyboardInterrupt:
        print("Stopping backend...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
