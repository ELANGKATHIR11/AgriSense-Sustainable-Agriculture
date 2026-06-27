#!/usr/bin/env python3
"""
GPU-Accelerated ML Model Retraining Pipeline for AgriSense
Uses RTX 5060 via TensorFlow 2.20.0 for optimal performance
"""

import os
import sys
import json
import logging
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import psutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPURetrainingOrchestrator:
    """Orchestrates GPU-accelerated model retraining"""
    
    def __init__(self, backend_path: Path, skip_models: List[str] = None):
        self.backend_path = Path(backend_path)
        self.models_dir = self.backend_path / "models"
        self.skip_models = skip_models or []
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 GPU Retraining Orchestrator initialized")
        logger.info(f"Backend path: {self.backend_path}")
        logger.info(f"Models directory: {self.models_dir}")
    
    def check_gpu_availability(self) -> bool:
        """Verify GPU is available for training"""
        logger.info("🔍 Checking GPU availability...")
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"✅ GPU detected: {len(gpus)} device(s)")
                for gpu in gpus:
                    logger.info(f"   - {gpu}")
                return True
            else:
                logger.warning("⚠️  No GPU detected. Will use CPU (slower)")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking GPU: {e}")
            return False
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check available system resources"""
        logger.info("📊 Checking system resources...")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'disk_available_gb': disk.free / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_percent': disk.percent,
        }
        
        logger.info(f"   CPU Usage: {cpu_percent}%")
        logger.info(f"   Memory: {resources['memory_available_gb']:.1f}GB / {resources['memory_total_gb']:.1f}GB ({memory.percent}%)")
        logger.info(f"   Disk: {resources['disk_available_gb']:.1f}GB / {resources['disk_total_gb']:.1f}GB ({disk.percent}%)")
        
        return resources
    
    def run_training_script(self, script_name: str, script_path: Path, description: str) -> Dict[str, Any]:
        """Run a training script with GPU acceleration"""
        if script_name in self.skip_models:
            logger.info(f"⏭️  Skipping {description}")
            return {"skipped": True}
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 {description}")
        logger.info(f"{'='*70}")
        
        start = time.time()
        
        try:
            # Run script in WSL with GPU environment
            cmd = [
                'wsl', '-d', 'Ubuntu-24.04', '--',
                'bash', '-c',
                f'source ~/tf_gpu_env/bin/activate && cd /mnt/d/AGRISENSEFULL-STACK && python {script_path}'
            ]
            
            logger.info(f"Running: {' '.join(cmd[:5])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            elapsed = time.time() - start
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully in {elapsed/60:.1f} minutes")
                logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
                return {
                    "success": True,
                    "elapsed_seconds": elapsed,
                    "output": result.stdout
                }
            else:
                logger.error(f"❌ {description} failed")
                logger.error(f"Error: {result.stderr}")
                return {
                    "success": False,
                    "elapsed_seconds": elapsed,
                    "error": result.stderr
                }
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out (>2 hours)")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            logger.error(f"❌ Error running {description}: {e}")
            return {"success": False, "error": str(e)}
    
    def retrain_all_models(self) -> Dict[str, Any]:
        """Retrain all ML models with GPU acceleration"""
        self.start_time = datetime.now()
        
        logger.info("\n" + "🔥" * 35)
        logger.info("GPU-ACCELERATED ML RETRAINING PIPELINE")
        logger.info("🔥" * 35)
        
        # Checks
        self.check_gpu_availability()
        resources = self.check_system_resources()
        
        if resources['memory_available_gb'] < 4:
            logger.warning("⚠️  Low memory available (<4GB). Training may be slow.")
        
        # Training sequence (order matters - dependencies)
        training_tasks = [
            {
                'name': 'plant_health',
                'script': 'tools/development/training_scripts/train_plant_health_models_v2.py',
                'description': '🌱 Training Plant Health Models (Disease & Weed Detection)'
            },
            {
                'name': 'deep_learning',
                'script': 'tools/development/training_scripts/deep_learning_pipeline_v2.py',
                'description': '🧠 Training Deep Learning Models (TensorFlow CNNs)'
            },
            {
                'name': 'gpu_hybrid',
                'script': 'tools/development/training_scripts/gpu_hybrid_ai_trainer.py',
                'description': '⚡ Training Hybrid AI Models (GPU-Optimized Ensemble)'
            },
            {
                'name': 'tf_gpu',
                'script': 'tools/development/training_scripts/tf_gpu_trainer.py',
                'description': '🎯 Training TensorFlow GPU Models (Final Optimization)'
            },
        ]
        
        # Run trainings
        for task in training_tasks:
            result = self.run_training_script(
                task['name'],
                Path(task['script']),
                task['description']
            )
            self.results[task['name']] = result
        
        self.end_time = datetime.now()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print retraining summary"""
        logger.info("\n" + "="*70)
        logger.info("📊 RETRAINING SUMMARY")
        logger.info("="*70)
        
        total_time = (self.end_time - self.start_time).total_seconds() / 60
        successful = sum(1 for r in self.results.values() if r.get('success'))
        failed = sum(1 for r in self.results.values() if not r.get('success') and not r.get('skipped'))
        skipped = sum(1 for r in self.results.values() if r.get('skipped'))
        
        logger.info(f"\n✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏭️  Skipped: {skipped}")
        logger.info(f"⏱️  Total Time: {total_time:.1f} minutes")
        
        logger.info("\nDetailed Results:")
        for model_name, result in self.results.items():
            if result.get('skipped'):
                status = "⏭️  SKIPPED"
            elif result.get('success'):
                status = f"✅ SUCCESS ({result.get('elapsed_seconds', 0)/60:.1f}m)"
            else:
                status = f"❌ FAILED: {result.get('error', 'Unknown error')}"
            logger.info(f"  {model_name}: {status}")
        
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Verify model accuracy improvements")
        logger.info("  2. Test models via API endpoints (/docs)")
        logger.info("  3. Update production models if satisfied")
        logger.info("  4. Monitor inference latency with new models")
        
        logger.info("\n" + "="*70)
    
    def save_retraining_report(self, output_file: Path = None):
        """Save retraining report to JSON"""
        if not output_file:
            output_file = self.backend_path / f"retraining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_minutes": (self.end_time - self.start_time).total_seconds() / 60 if (self.start_time and self.end_time) else None,
            "results": {k: {**v, 'output': v.get('output', '')[:200]} for k, v in self.results.items()},
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📄 Retraining report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated ML Model Retraining")
    parser.add_argument('--backend-path', type=str, default='agrisense_app/backend',
                       help='Path to backend directory')
    parser.add_argument('--skip', type=str, nargs='+', default=[],
                       help='Models to skip (e.g., --skip plant_health deep_learning)')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report, do not retrain')
    
    args = parser.parse_args()
    
    orchestrator = GPURetrainingOrchestrator(args.backend_path, args.skip)
    
    if not args.report_only:
        results = orchestrator.retrain_all_models()
        orchestrator.save_retraining_report()
    else:
        orchestrator._print_summary()

if __name__ == '__main__':
    main()
