#!/usr/bin/env python3
"""
Intel Core Ultra 9 275HX NPU Device Detection
Checks available NPU devices and capabilities
"""

import sys
from pathlib import Path

def check_openvino_npu():
    """Check OpenVINO NPU support"""
    print("=" * 70)
    print("🔍 INTEL NPU DEVICE DETECTION")
    print("=" * 70)
    
    try:
        from openvino.runtime import Core
        
        print("\n✅ OpenVINO Runtime imported successfully")
        
        core = Core()
        devices = core.available_devices
        
        print(f"\n📊 Available OpenVINO devices: {len(devices)}")
        for device in devices:
            print(f"  • {device}")
            
            # Get device properties
            if device:
                try:
                    device_name = core.get_property(device, "FULL_DEVICE_NAME")
                    print(f"    Name: {device_name}")
                except:
                    pass
        
        # Check specifically for NPU
        npu_available = "NPU" in devices
        if npu_available:
            print(f"\n🎯 NPU DETECTED! Device: NPU")
            npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
            print(f"    Full Name: {npu_name}")
            print(f"    ✅ Ready for model inference acceleration")
        else:
            print(f"\n⚠️ NPU not detected in OpenVINO runtime")
            print(f"    Available devices: {', '.join(devices)}")
            print(f"    Note: NPU support requires Intel Core Ultra processors (Meteor Lake+)")
        
        return npu_available
        
    except ImportError as e:
        print(f"\n❌ OpenVINO not installed: {e}")
        print(f"    Run: pip install -r requirements-npu.txt")
        return False
    except Exception as e:
        print(f"\n❌ Error checking NPU: {e}")
        return False

def check_ipex():
    """Check Intel Extension for PyTorch"""
    print("\n" + "=" * 70)
    print("⚡ INTEL PYTORCH EXTENSION (IPEX)")
    print("=" * 70)
    
    try:
        import torch
        import intel_extension_for_pytorch as ipex
        
        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"✅ IPEX version: {ipex.__version__}")
        
        # Check XPU (Intel GPU) support
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"\n🎮 Intel XPU (GPU) detected")
            print(f"    Device count: {torch.xpu.device_count()}")
        else:
            print(f"\n📝 XPU not available (CPU mode)")
        
        return True
        
    except ImportError as e:
        print(f"\n⚠️ IPEX not installed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error checking IPEX: {e}")
        return False

def check_neural_compressor():
    """Check Intel Neural Compressor"""
    print("\n" + "=" * 70)
    print("🗜️ INTEL NEURAL COMPRESSOR")
    print("=" * 70)
    
    try:
        import neural_compressor
        print(f"\n✅ Neural Compressor version: {neural_compressor.__version__}")
        print(f"    Purpose: Model quantization for NPU efficiency")
        return True
        
    except ImportError as e:
        print(f"\n⚠️ Neural Compressor not installed: {e}")
        return False

def check_scikit_learn_intelex():
    """Check Intel scikit-learn acceleration"""
    print("\n" + "=" * 70)
    print("🚄 INTEL SCIKIT-LEARN ACCELERATION")
    print("=" * 70)
    
    try:
        from sklearnex import patch_sklearn
        import sklearn
        
        print(f"\n✅ scikit-learn version: {sklearn.__version__}")
        print(f"✅ Intel extension available")
        
        # Patch sklearn to use Intel oneDAL
        patch_sklearn()
        print(f"✅ Patched scikit-learn to use Intel oneDAL")
        print(f"    This accelerates: RandomForest, GradientBoosting, KMeans, etc.")
        
        return True
        
    except ImportError as e:
        print(f"\n⚠️ Intel scikit-learn extension not installed: {e}")
        return False

def check_cpu_info():
    """Display CPU information"""
    print("\n" + "=" * 70)
    print("💻 CPU INFORMATION")
    print("=" * 70)
    
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        
        print(f"\n  Brand: {info.get('brand_raw', 'Unknown')}")
        print(f"  Architecture: {info.get('arch', 'Unknown')}")
        print(f"  Cores: {info.get('count', 'Unknown')}")
        print(f"  Hz: {info.get('hz_advertised_friendly', 'Unknown')}")
        
        # Check for NPU in CPU flags
        flags = info.get('flags', [])
        print(f"\n  Total CPU flags: {len(flags)}")
        
        # Look for relevant instructions
        relevant_flags = ['avx', 'avx2', 'avx512', 'vnni', 'amx']
        found_flags = [f for f in relevant_flags if any(f in flag for flag in flags)]
        if found_flags:
            print(f"  Relevant instructions: {', '.join(found_flags)}")
        
    except ImportError:
        print(f"\n⚠️ cpuinfo not installed")
    except Exception as e:
        print(f"\n❌ Error getting CPU info: {e}")

def main():
    """Main detection routine"""
    print("\n🚀 AgriSense NPU Hardware Detection")
    print(f"    Intel Core Ultra 9 275HX Optimization")
    print(f"    Date: 2025-12-30\n")
    
    results = {
        'openvino_npu': check_openvino_npu(),
        'ipex': check_ipex(),
        'neural_compressor': check_neural_compressor(),
        'sklearn_intelex': check_scikit_learn_intelex()
    }
    
    check_cpu_info()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    ready_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n  Components ready: {ready_count}/{total_count}")
    
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {component}")
    
    if results['openvino_npu']:
        print("\n🎉 NPU IS READY FOR MODEL ACCELERATION!")
        print("\n📋 Recommended next steps:")
        print("  1. Run hardware benchmark: python tools/npu/benchmark_hardware.py")
        print("  2. Train optimized models: python tools/npu/train_npu_optimized.py")
        print("  3. Convert models to OpenVINO IR: python tools/npu/convert_to_openvino.py")
    else:
        print("\n⚠️ NPU not detected - using CPU/GPU fallback")
        print("   Models will still be optimized for Intel hardware")
    
    print("\n" + "=" * 70 + "\n")
    
    return ready_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
