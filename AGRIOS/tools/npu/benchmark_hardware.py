#!/usr/bin/env python3
"""
Hardware Benchmark Tool for Intel Core Ultra 9 275HX
Measures CPU, NPU, and inference performance
"""

import time
import numpy as np
import psutil
import platform
from pathlib import Path
from typing import Dict, List

print("🔧 Loading libraries...")

# Try Intel extensions
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    INTEL_SKLEARN = True
except:
    INTEL_SKLEARN = False

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except:
    IPEX_AVAILABLE = False

try:
    from openvino.runtime import Core
    import openvino as ov
    OPENVINO_AVAILABLE = True
except:
    OPENVINO_AVAILABLE = False

import torch
import cpuinfo


class HardwareBenchmark:
    """Benchmark Intel Core Ultra 9 275HX capabilities"""
    
    def __init__(self):
        self.results = {}
        self.cpu_info = cpuinfo.get_cpu_info()
        
    def print_system_info(self):
        """Display system information"""
        print("\n" + "=" * 70)
        print("💻 SYSTEM INFORMATION")
        print("=" * 70)
        
        print(f"\n🖥️ CPU:")
        print(f"   Brand: {self.cpu_info.get('brand_raw', 'Unknown')}")
        print(f"   Architecture: {self.cpu_info.get('arch', 'Unknown')}")
        print(f"   Physical Cores: {psutil.cpu_count(logical=False)}")
        print(f"   Logical Cores: {psutil.cpu_count(logical=True)}")
        print(f"   Max Frequency: {psutil.cpu_freq().max:.0f} MHz")
        
        print(f"\n💾 Memory:")
        mem = psutil.virtual_memory()
        print(f"   Total: {mem.total / (1024**3):.1f} GB")
        print(f"   Available: {mem.available / (1024**3):.1f} GB")
        
        print(f"\n🐍 Python:")
        print(f"   Version: {platform.python_version()}")
        print(f"   Implementation: {platform.python_implementation()}")
        
        print(f"\n⚡ Acceleration:")
        print(f"   Intel scikit-learn: {'✅ Enabled' if INTEL_SKLEARN else '❌ Disabled'}")
        print(f"   Intel PyTorch (IPEX): {'✅ Enabled' if IPEX_AVAILABLE else '❌ Disabled'}")
        print(f"   OpenVINO: {'✅ Enabled' if OPENVINO_AVAILABLE else '❌ Disabled'}")
        
        if OPENVINO_AVAILABLE:
            core = Core()
            devices = core.available_devices
            print(f"\n🎯 OpenVINO Devices:")
            for device in devices:
                device_name = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"   • {device}: {device_name}")
    
    def benchmark_cpu_compute(self):
        """Benchmark CPU compute performance"""
        print("\n" + "=" * 70)
        print("⚙️ CPU COMPUTE BENCHMARK")
        print("=" * 70)
        
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            print(f"\n📊 Matrix size: {size}x{size}")
            
            # Generate random matrices
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # Matrix multiplication
            start = time.time()
            C = np.dot(A, B)
            duration = time.time() - start
            
            gflops = (2 * size**3) / (duration * 1e9)
            
            print(f"   Time: {duration:.3f}s")
            print(f"   Performance: {gflops:.2f} GFLOPS")
            
            results[f"matrix_{size}"] = {
                'time': duration,
                'gflops': gflops
            }
        
        self.results['cpu_compute'] = results
    
    def benchmark_sklearn_models(self):
        """Benchmark scikit-learn model training"""
        print("\n" + "=" * 70)
        print("🌲 SCIKIT-LEARN BENCHMARK")
        print("=" * 70)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Generate synthetic dataset
        print("\n📊 Generating dataset...")
        X, y = make_classification(
            n_samples=50000,
            n_features=20,
            n_informative=15,
            n_classes=10,
            random_state=42
        )
        
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        
        start = time.time()
        model.fit(X, y)
        train_time = time.time() - start
        
        accuracy = model.score(X, y)
        
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Training accuracy: {accuracy:.4f}")
        
        if INTEL_SKLEARN:
            print(f"   ✅ Accelerated with Intel oneDAL")
        
        self.results['sklearn_rf'] = {
            'train_time': train_time,
            'accuracy': accuracy,
            'intel_accelerated': INTEL_SKLEARN
        }
    
    def benchmark_pytorch_inference(self):
        """Benchmark PyTorch inference"""
        print("\n" + "=" * 70)
        print("🧠 PYTORCH INFERENCE BENCHMARK")
        print("=" * 70)
        
        # Create simple model
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(100, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 10)
                )
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleNet()
        model.eval()
        
        # Optimize with IPEX if available
        if IPEX_AVAILABLE:
            model = ipex.optimize(model)
            print("   ✅ Model optimized with IPEX")
        
        # Benchmark inference
        batch_size = 32
        num_iterations = 1000
        
        print(f"\n⏱️ Running {num_iterations} iterations (batch size: {batch_size})...")
        
        x = torch.randn(batch_size, 100)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        duration = time.time() - start
        
        throughput = (num_iterations * batch_size) / duration
        latency = (duration / num_iterations) * 1000  # ms
        
        print(f"   Total time: {duration:.2f}s")
        print(f"   Throughput: {throughput:.0f} samples/sec")
        print(f"   Latency: {latency:.2f} ms/batch")
        
        self.results['pytorch_inference'] = {
            'duration': duration,
            'throughput': throughput,
            'latency': latency,
            'ipex_optimized': IPEX_AVAILABLE
        }
    
    def benchmark_openvino_inference(self):
        """Benchmark OpenVINO inference on different devices"""
        if not OPENVINO_AVAILABLE:
            print("\n⚠️ OpenVINO not available, skipping...")
            return
        
        print("\n" + "=" * 70)
        print("🎯 OPENVINO INFERENCE BENCHMARK")
        print("=" * 70)
        
        core = Core()
        devices = core.available_devices
        
        # Create dummy IR model (just for benchmarking)
        print("\n📦 Creating test model...")
        
        # Simple model for testing
        import torch
        
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)
            
            def forward(self, x):
                return self.fc(x)
        
        dummy_model = DummyModel()
        dummy_input = torch.randn(1, 10)
        
        # Export to ONNX
        onnx_path = Path("temp_model.onnx")
        torch.onnx.export(
            dummy_model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output']
        )
        
        # Convert to OpenVINO
        from openvino.tools import mo
        ov_model = mo.convert_model(onnx_path)
        
        # Benchmark on each device
        results = {}
        
        for device in devices:
            print(f"\n🔹 Device: {device}")
            
            try:
                compiled_model = core.compile_model(ov_model, device)
                infer_request = compiled_model.create_infer_request()
                
                # Prepare input
                input_tensor = np.random.randn(1, 10).astype(np.float32)
                
                # Warmup
                for _ in range(10):
                    infer_request.infer({0: input_tensor})
                
                # Benchmark
                num_iterations = 1000
                start = time.time()
                for _ in range(num_iterations):
                    infer_request.infer({0: input_tensor})
                duration = time.time() - start
                
                throughput = num_iterations / duration
                latency = (duration / num_iterations) * 1000  # ms
                
                print(f"   Throughput: {throughput:.0f} inferences/sec")
                print(f"   Latency: {latency:.2f} ms")
                
                results[device] = {
                    'throughput': throughput,
                    'latency': latency
                }
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        
        # Cleanup
        onnx_path.unlink()
        
        self.results['openvino_inference'] = results
    
    def save_results(self):
        """Save benchmark results"""
        import json
        
        output_path = Path("npu_benchmark_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 70)
        
        if 'cpu_compute' in self.results:
            print("\n⚙️ CPU Compute:")
            for key, val in self.results['cpu_compute'].items():
                print(f"   {key}: {val['gflops']:.2f} GFLOPS")
        
        if 'sklearn_rf' in self.results:
            print("\n🌲 Scikit-learn Random Forest:")
            print(f"   Training time: {self.results['sklearn_rf']['train_time']:.2f}s")
            print(f"   Accuracy: {self.results['sklearn_rf']['accuracy']:.4f}")
        
        if 'pytorch_inference' in self.results:
            print("\n🧠 PyTorch Inference:")
            print(f"   Throughput: {self.results['pytorch_inference']['throughput']:.0f} samples/sec")
            print(f"   Latency: {self.results['pytorch_inference']['latency']:.2f} ms/batch")
        
        if 'openvino_inference' in self.results:
            print("\n🎯 OpenVINO Inference:")
            for device, metrics in self.results['openvino_inference'].items():
                print(f"   {device}:")
                print(f"      Throughput: {metrics['throughput']:.0f} inferences/sec")
                print(f"      Latency: {metrics['latency']:.2f} ms")
        
        print("\n" + "=" * 70)


def main():
    """Main benchmark routine"""
    print("\n" + "=" * 70)
    print("🚀 INTEL CORE ULTRA 9 275HX BENCHMARK")
    print("   AgriSense Hardware Performance Analysis")
    print("   Date: 2025-12-30")
    print("=" * 70)
    
    benchmark = HardwareBenchmark()
    
    # System info
    benchmark.print_system_info()
    
    # Run benchmarks
    print("\n" + "=" * 70)
    print("🎯 RUNNING BENCHMARKS")
    print("=" * 70)
    
    benchmark.benchmark_cpu_compute()
    benchmark.benchmark_sklearn_models()
    benchmark.benchmark_pytorch_inference()
    benchmark.benchmark_openvino_inference()
    
    # Results
    benchmark.save_results()
    benchmark.print_summary()
    
    print("\n✅ BENCHMARK COMPLETE!")
    print("\n💡 Tips:")
    print("   - Higher GFLOPS = Better CPU performance")
    print("   - Higher throughput = Better inference speed")
    print("   - Lower latency = Faster response time")
    print("   - NPU should show lowest latency for small models\n")


if __name__ == "__main__":
    main()
