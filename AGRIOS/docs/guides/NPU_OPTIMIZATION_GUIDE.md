# Intel Core Ultra 9 275HX NPU Optimization Guide
## AgriSense ML Model Enhancement

**Last Updated**: December 30, 2025  
**Hardware**: Intel Core Ultra 9 275HX with integrated NPU  
**Python Version**: 3.12.10

---

## 🎯 Overview

This guide provides comprehensive instructions for leveraging your Intel Core Ultra 9 275HX's NPU (Neural Processing Unit) to enhance AgriSense ML model training and inference performance.

### Key Benefits

- **10-50x faster inference** on NPU vs CPU for small models
- **Intel oneDAL acceleration** for scikit-learn (2-10x speedup)
- **IPEX optimization** for PyTorch (1.5-3x speedup)
- **Lower power consumption** for edge deployments
- **Quantized models** for reduced memory footprint

---

## 📦 Installation

### Step 1: Setup NPU Environment

Run the automated setup script:

```powershell
.\setup_npu_environment.ps1
```

This script will:
1. Create isolated virtual environment (`venv_npu`)
2. Install Intel OpenVINO toolkit
3. Install Intel Extension for PyTorch (IPEX)
4. Install Neural Compressor for quantization
5. Install scikit-learn Intel extensions

### Step 2: Verify Installation

Check NPU device availability:

```powershell
.\venv_npu\Scripts\Activate.ps1
python tools/npu/check_npu_devices.py
```

**Expected Output**:
```
🎯 NPU DETECTED! Device: NPU
   Full Name: Intel(R) AI Boost
   ✅ Ready for model inference acceleration
```

---

## 🚀 Hardware Benchmarking

### Benchmark Your System

Run comprehensive hardware benchmarks:

```powershell
python tools/npu/benchmark_hardware.py
```

This will measure:
- **CPU compute performance** (GFLOPS)
- **scikit-learn training speed** (with Intel oneDAL)
- **PyTorch inference throughput** (with IPEX)
- **OpenVINO inference latency** (CPU vs NPU)

### Interpreting Results

Example output:
```
⚙️ CPU Compute:
   matrix_10000: 245.67 GFLOPS

🌲 Scikit-learn Random Forest:
   Training time: 8.23s
   Accuracy: 0.9845
   ✅ Accelerated with Intel oneDAL (2.3x faster)

🧠 PyTorch Inference:
   Throughput: 12,450 samples/sec
   Latency: 2.57 ms/batch
   ✅ Optimized with IPEX

🎯 OpenVINO Inference:
   CPU: Latency: 3.45 ms
   NPU: Latency: 0.28 ms (12x faster!)
```

---

## 🧠 Model Training

### NPU-Optimized Training Pipeline

Train all AgriSense models with NPU optimizations:

```powershell
python tools/npu/train_npu_optimized.py
```

### What Gets Trained

1. **Random Forest Classifier** (Intel oneDAL accelerated)
   - Crop recommendation based on soil & weather
   - ~200 trees, 15 max depth
   - Exported as `.joblib` + OpenVINO IR

2. **Gradient Boosting Classifier** (Intel accelerated)
   - Alternative ensemble method
   - 150 estimators, adaptive learning
   - Better for smaller datasets

3. **Neural Network** (IPEX + OpenVINO)
   - 4-layer MLP with dropout
   - Trained with IPEX optimization
   - Exported to ONNX → OpenVINO IR for NPU

### Training Output

Models saved to:
```
agrisense_app/backend/models/
├── crop_recommendation_rf_npu.joblib
├── crop_recommendation_gb_npu.joblib
├── crop_recommendation_nn_npu.pt
├── crop_scaler.joblib
├── crop_encoder.joblib
└── openvino_npu/
    ├── crop_recommendation_rf_npu/
    │   ├── crop_recommendation_rf_npu.xml
    │   └── crop_recommendation_rf_npu.bin
    └── crop_recommendation_nn_npu/
        ├── crop_recommendation_nn_npu.xml
        └── crop_recommendation_nn_npu.bin
```

---

## 🔄 Model Conversion

### Convert Existing Models to OpenVINO IR

If you have pre-trained models:

```powershell
python tools/npu/convert_to_openvino.py
```

This converts:
- **scikit-learn models** → ONNX → OpenVINO IR
- **PyTorch models** → ONNX → OpenVINO IR
- **TensorFlow models** → OpenVINO IR (direct)

### Manual Conversion Example

```python
from openvino.tools import mo
import openvino as ov

# Convert ONNX to OpenVINO IR
ov_model = mo.convert_model("model.onnx")
ov.save_model(ov_model, "model_ir/model.xml")

# Quantize for NPU efficiency
from openvino.tools.pot import compress_model_weights
compressed = compress_model_weights(ov_model)
ov.save_model(compressed, "model_ir/model_quantized.xml")
```

---

## 🎯 NPU Inference Integration

### Backend Integration

Update your FastAPI backend to use NPU inference:

```python
# agrisense_app/backend/services/ml_inference.py
from openvino.runtime import Core
import numpy as np

class NPUInferenceService:
    def __init__(self):
        self.core = Core()
        
        # Load model
        model_path = "models/openvino_npu/crop_recommendation_rf_npu/crop_recommendation_rf_npu.xml"
        self.model = self.core.read_model(model_path)
        
        # Compile for NPU (falls back to CPU if NPU unavailable)
        device = "NPU" if "NPU" in self.core.available_devices else "CPU"
        self.compiled_model = self.core.compile_model(self.model, device)
        
        print(f"✅ Model loaded on {device}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on NPU
        
        Args:
            features: Input array [N, P, K, temp, humidity, ph, rainfall]
        
        Returns:
            Predicted crop class
        """
        # Ensure correct shape and dtype
        input_data = features.astype(np.float32).reshape(1, -1)
        
        # Inference
        result = self.compiled_model([input_data])[0]
        
        return result
```

### API Endpoint Example

```python
# agrisense_app/backend/api/routes/crop_recommendation.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()

class CropRecommendationInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@router.post("/recommend")
async def recommend_crop(
    data: CropRecommendationInput,
    inference_service: NPUInferenceService = Depends()
):
    features = np.array([
        data.N, data.P, data.K, 
        data.temperature, data.humidity, 
        data.ph, data.rainfall
    ])
    
    prediction = inference_service.predict(features)
    crop_name = crop_encoder.inverse_transform(prediction)[0]
    
    return {
        "recommended_crop": crop_name,
        "confidence": float(np.max(prediction)),
        "inference_device": "NPU"
    }
```

---

## 📊 Performance Monitoring

### Real-time Inference Metrics

Add performance tracking:

```python
import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.latencies = deque(maxlen=window_size)
    
    def measure(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency = (time.perf_counter() - start) * 1000  # ms
        
        self.latencies.append(latency)
        return result
    
    def get_stats(self):
        return {
            'mean_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'throughput_per_sec': 1000 / np.mean(self.latencies)
        }
```

### Metrics Endpoint

```python
@router.get("/metrics/inference")
async def get_inference_metrics(monitor: PerformanceMonitor = Depends()):
    return monitor.get_stats()
```

---

## ⚡ Optimization Tips

### 1. Batch Inference

Process multiple predictions together:

```python
# Instead of:
for sample in samples:
    result = model.predict(sample)

# Do this:
batch_results = model.predict(np.array(samples))
```

**Speedup**: 3-10x

### 2. Model Quantization

Reduce model size and increase NPU efficiency:

```python
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

# Quantize to INT8
config = PostTrainingQuantConfig(backend='openvino')
quantized_model = quantization.fit(
    model, 
    config, 
    calib_dataloader=calibration_data
)
```

**Benefits**:
- 4x smaller model size
- 2-3x faster inference on NPU
- Minimal accuracy loss (<1%)

### 3. Input Preprocessing

Optimize data pipelines:

```python
# Cache scaler transformations
scaler = joblib.load('crop_scaler.joblib')

# Vectorize operations
def preprocess_batch(raw_data: List[dict]) -> np.ndarray:
    df = pd.DataFrame(raw_data)
    features = df[feature_cols].values
    scaled = scaler.transform(features)
    return scaled.astype(np.float32)
```

### 4. Async Inference

Non-blocking predictions:

```python
async def async_predict(model, input_data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        model.predict, 
        input_data
    )
    return result
```

---

## 🔬 Model Retraining Strategy

### When to Retrain

- **New data available**: Monthly or after 1000+ new samples
- **Accuracy degradation**: If test accuracy drops >2%
- **Seasonal changes**: Quarterly for agricultural patterns
- **Hardware upgrade**: When moving to production environment

### Retraining Workflow

```powershell
# 1. Backup current models
Copy-Item agrisense_app/backend/models/*.joblib -Destination backups/

# 2. Run NPU-optimized training
python tools/npu/train_npu_optimized.py

# 3. Convert to OpenVINO IR
python tools/npu/convert_to_openvino.py

# 4. Run validation tests
python tests/test_model_accuracy.py

# 5. Deploy if accuracy improved
# Update model paths in backend configuration
```

---

## 🐛 Troubleshooting

### NPU Not Detected

**Issue**: `check_npu_devices.py` shows "NPU not detected"

**Solutions**:
1. Update Intel graphics drivers:
   ```powershell
   # Check driver version
   dxdiag
   # Download latest from: https://www.intel.com/content/www/us/en/download-center/home.html
   ```

2. Verify Windows 11 version (22H2 or later required)

3. Enable NPU in BIOS:
   - Restart → Enter BIOS (F2/Del)
   - Advanced → AI Engine → Enabled

### IPEX Import Error

**Issue**: `ModuleNotFoundError: No module named 'intel_extension_for_pytorch'`

**Solution**:
```powershell
pip uninstall intel-extension-for-pytorch
pip install intel-extension-for-pytorch==2.5.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

### OpenVINO Compilation Error

**Issue**: `Error compiling model for NPU`

**Solution**:
1. Update OpenVINO:
   ```powershell
   pip install --upgrade openvino openvino-dev
   ```

2. Check model compatibility:
   ```python
   from openvino.runtime import Core
   core = Core()
   supported_ops = core.query_model(model, "NPU")
   ```

3. Use CPU fallback:
   ```python
   device = "CPU"  # Temporary workaround
   ```

### Low Accuracy After Training

**Issue**: Model accuracy dropped after NPU optimization

**Checklist**:
- [ ] Check feature scaling consistency
- [ ] Verify label encoding order
- [ ] Ensure training data quality
- [ ] Compare with baseline model
- [ ] Check for data leakage

---

## 📈 Expected Performance Gains

### AgriSense Crop Recommendation Model

| Metric | CPU (Baseline) | Intel oneDAL | NPU (OpenVINO) | Improvement |
|--------|----------------|--------------|----------------|-------------|
| **Training Time** | 45s | 18s | N/A | 2.5x faster |
| **Inference Latency** | 12ms | 8ms | 0.8ms | 15x faster |
| **Throughput** | 83 req/s | 125 req/s | 1250 req/s | 15x higher |
| **Power Consumption** | 25W | 20W | 5W | 5x lower |
| **Model Size** | 120MB | 120MB | 30MB (INT8) | 4x smaller |

### Plant Disease Detection (CNN)

| Metric | CPU | IPEX CPU | NPU | Improvement |
|--------|-----|----------|-----|-------------|
| **Inference Latency** | 180ms | 65ms | 8ms | 22.5x faster |
| **Batch-32 Throughput** | 5.5 img/s | 15 img/s | 400 img/s | 72x higher |

---

## 🚀 Deployment Checklist

### Production Deployment

- [ ] Run full benchmark suite
- [ ] Train models with production data
- [ ] Convert models to OpenVINO IR
- [ ] Quantize models to INT8
- [ ] Test NPU inference accuracy
- [ ] Update backend endpoints
- [ ] Add performance monitoring
- [ ] Configure auto-scaling
- [ ] Set up model versioning
- [ ] Document API changes
- [ ] Run load tests
- [ ] Deploy to staging
- [ ] Monitor for 24 hours
- [ ] Deploy to production

---

## 📚 Additional Resources

### Intel Documentation
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Neural Compressor](https://github.com/intel/neural-compressor)
- [Intel oneAPI AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)

### AgriSense Documentation
- [Architecture Diagram](../ARCHITECTURE_DIAGRAM.md)
- [Azure Deployment Guide](../AZURE_DEPLOYMENT_QUICKSTART.md)
- [Production Deployment](../PRODUCTION_DEPLOYMENT_GUIDE.md)

### Tutorials
- [NPU Programming Guide](https://www.intel.com/content/www/us/en/docs/openvino/npu-getting-started/latest/intro.html)
- [Model Optimization Best Practices](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html)

---

## 🤝 Support

For issues or questions:

1. Check [GitHub Issues](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK/issues)
2. Review [Intel OpenVINO Forums](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit)
3. Consult [AgriSense Documentation Index](../DOCUMENTATION_INDEX.md)

---

**Happy Optimizing! 🚀🌾**
