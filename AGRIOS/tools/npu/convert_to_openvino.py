#!/usr/bin/env python3
"""
Model Converter: Export trained models to OpenVINO IR for NPU inference
Converts PyTorch and TensorFlow models to NPU-optimized format
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

try:
    import openvino as ov
    from openvino.tools import mo
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    print("❌ OpenVINO not installed. Run: pip install -r requirements-npu.txt")
    sys.exit(1)


class ModelConverter:
    """Convert ML models to OpenVINO IR format"""
    
    def __init__(self, models_dir: str = "agrisense_app/backend/models"):
        self.models_dir = Path(models_dir)
        self.output_dir = self.models_dir / "openvino_ir"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Input models: {self.models_dir}")
        print(f"📁 Output IR: {self.output_dir}")
        
        self.core = Core()
        devices = self.core.available_devices
        self.npu_available = "NPU" in devices
        
        print(f"\n🎯 NPU Status: {'✅ Available' if self.npu_available else '⚠️ Not detected'}")
    
    def convert_pytorch_to_ir(self, model: torch.nn.Module, model_name: str, input_shape: tuple):
        """Convert PyTorch model to OpenVINO IR"""
        print(f"\n{'=' * 70}")
        print(f"🔄 Converting PyTorch model: {model_name}")
        print(f"{'=' * 70}")
        
        # Set to eval mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        print(f"\n1️⃣ Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"   ✅ ONNX: {onnx_path}")
        
        # Convert ONNX to OpenVINO IR
        print(f"\n2️⃣ Converting to OpenVINO IR...")
        ir_path = self.output_dir / model_name
        ir_path.mkdir(exist_ok=True)
        
        ov_model = mo.convert_model(onnx_path)
        ov.save_model(ov_model, str(ir_path / f"{model_name}.xml"))
        
        print(f"   ✅ IR: {ir_path}/{model_name}.xml")
        print(f"   ✅ Weights: {ir_path}/{model_name}.bin")
        
        # Test inference
        self.test_inference(ir_path / f"{model_name}.xml", dummy_input.numpy())
        
        return ir_path
    
    def convert_sklearn_to_ir(self, model_path: Path, model_name: str, input_dim: int):
        """Convert scikit-learn model to OpenVINO IR via ONNX"""
        print(f"\n{'=' * 70}")
        print(f"🔄 Converting scikit-learn model: {model_name}")
        print(f"{'=' * 70}")
        
        try:
            import joblib
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Load sklearn model
            print(f"\n1️⃣ Loading model: {model_path}")
            model = joblib.load(model_path)
            
            # Convert to ONNX
            print(f"\n2️⃣ Converting to ONNX...")
            initial_type = [('float_input', FloatTensorType([None, input_dim]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
            
            onnx_path = self.output_dir / f"{model_name}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"   ✅ ONNX: {onnx_path}")
            
            # Convert to OpenVINO IR
            print(f"\n3️⃣ Converting to OpenVINO IR...")
            ir_path = self.output_dir / model_name
            ir_path.mkdir(exist_ok=True)
            
            ov_model = mo.convert_model(onnx_path)
            ov.save_model(ov_model, str(ir_path / f"{model_name}.xml"))
            
            print(f"   ✅ IR: {ir_path}/{model_name}.xml")
            
            # Test inference
            dummy_input = np.random.randn(1, input_dim).astype(np.float32)
            self.test_inference(ir_path / f"{model_name}.xml", dummy_input)
            
            return ir_path
            
        except ImportError:
            print("   ⚠️ skl2onnx not installed. Run: pip install skl2onnx")
            return None
        except Exception as e:
            print(f"   ❌ Conversion failed: {e}")
            return None
    
    def test_inference(self, ir_path: Path, dummy_input: np.ndarray):
        """Test OpenVINO model inference on available devices"""
        print(f"\n4️⃣ Testing inference...")
        
        model = self.core.read_model(ir_path)
        
        # Test on CPU
        print(f"   Testing on CPU...")
        compiled_model = self.core.compile_model(model, "CPU")
        output = compiled_model([dummy_input])[0]
        print(f"   ✅ CPU inference successful - Output shape: {output.shape}")
        
        # Test on NPU if available
        if self.npu_available:
            try:
                print(f"   Testing on NPU...")
                compiled_model_npu = self.core.compile_model(model, "NPU")
                output_npu = compiled_model_npu([dummy_input])[0]
                print(f"   ✅ NPU inference successful - Output shape: {output_npu.shape}")
            except Exception as e:
                print(f"   ⚠️ NPU inference failed: {e}")
    
    def optimize_for_npu(self, ir_path: Path):
        """Apply NPU-specific optimizations"""
        print(f"\n5️⃣ Applying NPU optimizations...")
        
        if not self.npu_available:
            print(f"   ⚠️ NPU not available, skipping...")
            return
        
        try:
            # Quantization to INT8 for NPU efficiency
            from openvino.tools.pot import compress_model_weights
            
            model = self.core.read_model(ir_path)
            compressed_model = compress_model_weights(model)
            
            optimized_path = ir_path.parent / f"{ir_path.stem}_npu_optimized.xml"
            ov.save_model(compressed_model, str(optimized_path))
            
            print(f"   ✅ Optimized model: {optimized_path}")
            
        except Exception as e:
            print(f"   ⚠️ Optimization failed: {e}")


def main():
    """Main conversion pipeline"""
    print("\n" + "=" * 70)
    print("🔄 AGRISENSE MODEL CONVERSION TO OPENVINO IR")
    print("   NPU-Optimized Inference Format")
    print("   Date: 2025-12-30")
    print("=" * 70)
    
    converter = ModelConverter()
    
    # Convert trained models
    models_to_convert = [
        # Format: (model_path, model_name, input_dim, model_type)
        ("agrisense_app/backend/models/crop_recommendation_rf_npu.joblib", 
         "crop_rf_npu", 7, "sklearn"),
        
        ("agrisense_app/backend/models/crop_recommendation_gb_npu.joblib",
         "crop_gb_npu", 7, "sklearn"),
    ]
    
    print("\n📦 Models to convert:")
    for model_path, model_name, _, _ in models_to_convert:
        status = "✅" if Path(model_path).exists() else "❌"
        print(f"   {status} {model_path}")
    
    # Convert each model
    for model_path, model_name, input_dim, model_type in models_to_convert:
        if not Path(model_path).exists():
            print(f"\n⚠️ Skipping {model_path} (not found)")
            continue
        
        if model_type == "sklearn":
            ir_path = converter.convert_sklearn_to_ir(
                Path(model_path), model_name, input_dim
            )
            
            if ir_path and converter.npu_available:
                converter.optimize_for_npu(ir_path / f"{model_name}.xml")
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    
    if converter.npu_available:
        print("\n🎯 NPU-optimized models ready for deployment!")
        print(f"\n📁 Models location: {converter.output_dir}")
        print("\n💡 Integration steps:")
        print("   1. Update backend to use OpenVINO inference")
        print("   2. Load models with: core.read_model('path/to/model.xml')")
        print("   3. Compile for NPU: core.compile_model(model, 'NPU')")
        print("   4. Run inference: compiled_model([input_data])")
    else:
        print("\n⚠️ NPU not available - models optimized for CPU")
    
    print()


if __name__ == "__main__":
    main()
