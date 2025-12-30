import subprocess
import sys
import importlib

VERSIONS = [
    "2.18.0",
    "2.17.1",
    "2.16.2",
    "2.16.1",
    "2.15.1",
    "2.9.1",
]

INDEX_URL = "https://download.pytorch.org/whl/cu131"

def try_install(version):
    pkg = f"torch=={version}+cu131"
    print(f"Trying install: {pkg}")
    res = subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-f", INDEX_URL, "--no-cache-dir"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        return False
    try:
        torch = importlib.import_module('torch')
        cuda_avail = getattr(torch, 'cuda', None) and torch.cuda.is_available()
        print('torch', torch.__version__, 'cuda_available=', cuda_avail)
        return bool(cuda_avail)
    except Exception as e:
        print('Import check failed:', e)
        return False

def main():
    for v in VERSIONS:
        ok = try_install(v)
        if ok:
            print('SUCCESS: CUDA-enabled torch installed')
            return 0
    print('No candidate succeeded; consider manually choosing a wheel matching your CUDA.')
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
