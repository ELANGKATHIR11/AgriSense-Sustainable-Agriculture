
import sys
import os
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "AgriSense"))

print(f"Project root: {project_root}")
print(f"sys.path: {sys.path}")

try:
    import agrisense_app
    print("Successfully imported agrisense_app")
except ImportError as e:
    print(f"Failed to import agrisense_app: {e}")

try:
    from main import app
    print("Successfully loaded app from main")
    
    # Check routes
    print("Routes:")
    for route in app.routes:
        print(f"  {route.path} [{route.name}]")
        
except Exception as e:
    print(f"Failed to load app from main: {e}")
