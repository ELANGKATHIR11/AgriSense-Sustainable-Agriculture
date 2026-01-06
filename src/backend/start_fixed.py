
import sys
import os
from pathlib import Path
import uvicorn

# Add src to sys.path so we can import backend
src_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_path))

# Also add the current directory (src/backend) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Mock agrisense_app package structure
import types
agrisense_app = types.ModuleType("agrisense_app")
sys.modules["agrisense_app"] = agrisense_app

# Import backend from src and alias it to agrisense_app.backend
import backend
agrisense_app.backend = backend
sys.modules["agrisense_app.backend"] = backend

# Now we need to make sure submodules are also aliased if needed
# But since backend is a package, importing agrisense_app.backend.core should work 
# if backend.core exists.

# Let's verify
try:
    import agrisense_app.backend.core
    print("✅ Successfully mocked agrisense_app.backend.core")
except ImportError as e:
    print(f"⚠️  Could not import agrisense_app.backend.core: {e}")
    # Try to manually alias core if it fails
    try:
        import backend.core
        agrisense_app.backend.core = backend.core
        sys.modules["agrisense_app.backend.core"] = backend.core
        print("✅ Manually aliased agrisense_app.backend.core")
    except ImportError as e2:
        print(f"❌ Failed to import backend.core: {e2}")

if __name__ == "__main__":
    print("🚀 Starting AgriSense Backend with fixed imports...")
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=False, log_level="debug")
    except Exception as e:
        print(f"❌ Uvicorn crashed: {e}")
    except KeyboardInterrupt:
        print("⚠️ KeyboardInterrupt detected")

