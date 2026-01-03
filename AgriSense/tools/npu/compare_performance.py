#!/usr/bin/env python3
"""
Performance Comparison Tool
Compare CPU vs NPU-optimized models
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except:
    OPENVINO_AVAILABLE = False

import joblib


class PerformanceComparator:
    """Compare CPU and NPU model performance"""
    
    def __init__(self):
        self.results = {
            'cpu': {},
            'npu': {},
            'comparison': {}
        }
        
        if OPENVINO_AVAILABLE:
            self.core = Core()
            self.npu_available = "NPU" in self.core.available_devices
        else:
            self.npu_available = False
        
        print(f"🎯 NPU Available: {self.npu_available}")
    
    def load_sklearn_model(self, model_path: Path):
        """Load scikit-learn model"""
        return joblib.load(model_path)
    
    def load_openvino_model(self, model_path: Path, device: str = "CPU"):
        """Load OpenVINO IR model"""
        if not OPENVINO_AVAILABLE:
            return None
        
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, device)
        return compiled_model
    
    def benchmark_sklearn_inference(
        self, 
        model, 
        test_data: np.ndarray, 
        num_iterations: int = 1000
    ) -> Dict:
        """Benchmark scikit-learn model"""
        print(f"\n⏱️ Benchmarking scikit-learn (CPU)...")
        
        # Warmup
        for _ in range(10):
            _ = model.predict(test_data[:1])
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model.predict(test_data[:1])
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_per_sec': 1000 / np.mean(latencies)
        }
    
    def benchmark_openvino_inference(
        self,
        model,
        test_data: np.ndarray,
        num_iterations: int = 1000,
        device: str = "CPU"
    ) -> Dict:
        """Benchmark OpenVINO model"""
        print(f"\n⏱️ Benchmarking OpenVINO ({device})...")
        
        if model is None:
            return None
        
        # Prepare input
        input_data = test_data[:1].astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = model([input_data])
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model([input_data])
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_per_sec': 1000 / np.mean(latencies)
        }
    
    def compare_models(self, model_name: str = "crop_recommendation_rf_npu"):
        """Compare CPU vs NPU performance"""
        print(f"\n{'=' * 70}")
        print(f"🔬 COMPARING: {model_name}")
        print(f"{'=' * 70}")
        
        models_dir = Path("agrisense_app/backend/models")
        
        # Generate test data
        test_data = np.random.randn(100, 7).astype(np.float32)
        
        # Load and benchmark sklearn model (CPU)
        sklearn_path = models_dir / f"{model_name}.joblib"
        if sklearn_path.exists():
            sklearn_model = self.load_sklearn_model(sklearn_path)
            cpu_results = self.benchmark_sklearn_inference(sklearn_model, test_data)
            self.results['cpu'][model_name] = cpu_results
        else:
            print(f"⚠️ sklearn model not found: {sklearn_path}")
            cpu_results = None
        
        # Load and benchmark OpenVINO model (CPU)
        openvino_cpu_path = models_dir / f"openvino_npu/{model_name}/{model_name}.xml"
        if OPENVINO_AVAILABLE and openvino_cpu_path.exists():
            ov_model_cpu = self.load_openvino_model(openvino_cpu_path, "CPU")
            ov_cpu_results = self.benchmark_openvino_inference(ov_model_cpu, test_data, device="CPU")
            self.results['cpu'][f"{model_name}_openvino"] = ov_cpu_results
        
        # Load and benchmark OpenVINO model (NPU)
        if self.npu_available and openvino_cpu_path.exists():
            ov_model_npu = self.load_openvino_model(openvino_cpu_path, "NPU")
            npu_results = self.benchmark_openvino_inference(ov_model_npu, test_data, device="NPU")
            self.results['npu'][model_name] = npu_results
            
            # Calculate speedup
            if cpu_results:
                speedup = cpu_results['mean_latency_ms'] / npu_results['mean_latency_ms']
                throughput_increase = npu_results['throughput_per_sec'] / cpu_results['throughput_per_sec']
                
                self.results['comparison'][model_name] = {
                    'latency_speedup': speedup,
                    'throughput_increase': throughput_increase
                }
        
        # Print results
        self.print_comparison(model_name)
    
    def print_comparison(self, model_name: str):
        """Print comparison results"""
        print(f"\n📊 RESULTS FOR {model_name}")
        print("=" * 70)
        
        if model_name in self.results['cpu']:
            cpu_metrics = self.results['cpu'][model_name]
            print(f"\n🖥️ CPU (scikit-learn):")
            print(f"   Mean latency: {cpu_metrics['mean_latency_ms']:.2f} ms")
            print(f"   P95 latency: {cpu_metrics['p95_latency_ms']:.2f} ms")
            print(f"   Throughput: {cpu_metrics['throughput_per_sec']:.0f} req/s")
        
        if f"{model_name}_openvino" in self.results['cpu']:
            ov_cpu_metrics = self.results['cpu'][f"{model_name}_openvino"]
            print(f"\n⚡ CPU (OpenVINO):")
            print(f"   Mean latency: {ov_cpu_metrics['mean_latency_ms']:.2f} ms")
            print(f"   P95 latency: {ov_cpu_metrics['p95_latency_ms']:.2f} ms")
            print(f"   Throughput: {ov_cpu_metrics['throughput_per_sec']:.0f} req/s")
        
        if model_name in self.results['npu']:
            npu_metrics = self.results['npu'][model_name]
            print(f"\n🎯 NPU (OpenVINO):")
            print(f"   Mean latency: {npu_metrics['mean_latency_ms']:.2f} ms")
            print(f"   P95 latency: {npu_metrics['p95_latency_ms']:.2f} ms")
            print(f"   Throughput: {npu_metrics['throughput_per_sec']:.0f} req/s")
        
        if model_name in self.results['comparison']:
            comp = self.results['comparison'][model_name]
            print(f"\n🚀 IMPROVEMENT:")
            print(f"   Latency speedup: {comp['latency_speedup']:.1f}x faster")
            print(f"   Throughput increase: {comp['throughput_increase']:.1f}x higher")
    
    def save_results(self):
        """Save results to JSON"""
        output_path = Path("npu_comparison_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
    
    def plot_comparison(self):
        """Plot comparison charts"""
        if not self.results['comparison']:
            print("\n⚠️ No comparison data to plot")
            return
        
        print("\n📊 Generating comparison charts...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        models = list(self.results['comparison'].keys())
        latency_speedups = [self.results['comparison'][m]['latency_speedup'] for m in models]
        throughput_increases = [self.results['comparison'][m]['throughput_increase'] for m in models]
        
        # Latency speedup
        ax1.barh(models, latency_speedups, color='green', alpha=0.7)
        ax1.set_xlabel('Speedup Factor (x)', fontsize=12)
        ax1.set_title('NPU Latency Speedup vs CPU', fontsize=14, fontweight='bold')
        ax1.axvline(x=1, color='red', linestyle='--', label='Baseline (CPU)')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Throughput increase
        ax2.barh(models, throughput_increases, color='blue', alpha=0.7)
        ax2.set_xlabel('Increase Factor (x)', fontsize=12)
        ax2.set_title('NPU Throughput Increase vs CPU', fontsize=14, fontweight='bold')
        ax2.axvline(x=1, color='red', linestyle='--', label='Baseline (CPU)')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = Path("npu_comparison_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Chart saved: {chart_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print overall summary"""
        print("\n" + "=" * 70)
        print("🏆 PERFORMANCE SUMMARY")
        print("=" * 70)
        
        if not self.results['comparison']:
            print("\n⚠️ No NPU comparisons available")
            return
        
        avg_latency_speedup = np.mean([
            c['latency_speedup'] 
            for c in self.results['comparison'].values()
        ])
        
        avg_throughput_increase = np.mean([
            c['throughput_increase']
            for c in self.results['comparison'].values()
        ])
        
        print(f"\n📊 Average Performance Gains:")
        print(f"   Latency speedup: {avg_latency_speedup:.1f}x faster")
        print(f"   Throughput increase: {avg_throughput_increase:.1f}x higher")
        
        print(f"\n💡 Interpretation:")
        if avg_latency_speedup > 10:
            print(f"   🎉 Excellent! NPU provides significant acceleration")
        elif avg_latency_speedup > 5:
            print(f"   ✅ Great! NPU provides substantial speedup")
        elif avg_latency_speedup > 2:
            print(f"   👍 Good! NPU provides noticeable improvement")
        else:
            print(f"   ⚠️ Modest gains - may need model optimization")
        
        print("\n" + "=" * 70)


def main():
    """Main comparison routine"""
    print("\n" + "=" * 70)
    print("🔬 AGRISENSE NPU PERFORMANCE COMPARISON")
    print("   Intel Core Ultra 9 275HX")
    print("   Date: 2025-12-30")
    print("=" * 70)
    
    comparator = PerformanceComparator()
    
    if not comparator.npu_available:
        print("\n⚠️ NPU not available - showing CPU-only benchmarks")
    
    # Compare models
    models_to_compare = [
        "crop_recommendation_rf_npu",
        "crop_recommendation_gb_npu"
    ]
    
    for model_name in models_to_compare:
        comparator.compare_models(model_name)
    
    # Save and plot results
    comparator.save_results()
    
    if comparator.npu_available:
        comparator.plot_comparison()
    
    comparator.print_summary()
    
    print("\n✅ COMPARISON COMPLETE!\n")


if __name__ == "__main__":
    main()
