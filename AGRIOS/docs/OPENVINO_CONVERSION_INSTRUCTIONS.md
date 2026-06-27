OpenVINO Conversion Quick Guide
================================

Prerequisites
-------------
- Use Python 3.12 environment (conda recommended).
- Install OpenVINO and dev tools:

```bash
conda create -n agrisense_openvino python=3.12 -y
conda activate agrisense_openvino
pip install --upgrade pip setuptools wheel
pip install openvino openvino-dev onnx onnxruntime skl2onnx
```

Typical workflow
-----------------
1. Export model to ONNX
   - PyTorch: `torch.onnx.export(model, dummy_input, "model.onnx", opset_version=16)`
   - scikit-learn: use `skl2onnx.convert_sklearn()` with an `initial_types` describing input shape

2. Convert ONNX -> OpenVINO IR

```bash
python -m openvino.tools.mo --input_model model.onnx --output_dir models/openvino_npu/
```

3. Validate using OpenVINO runtime

```python
from openvino.runtime import Core
core = Core()
model = core.read_model('models/openvino_npu/model.xml')
compiled = core.compile_model(model, device_name='CPU')
```

Notes
-----
- For NPU targeting, replace `device_name='CPU'` with the NPU device id once OpenVINO detects it (e.g., `AUTO:GPU,MYRIAD` or device name provided by `openvino.runtime.Core().available_devices`).
- For INT8 quantization, use `openvino.tools.pot` or `neural_compressor` to calibrate models.
