import traceback
import importlib

try:
    importlib.import_module('main')
    print('Imported main successfully')
except Exception:
    traceback.print_exc()
