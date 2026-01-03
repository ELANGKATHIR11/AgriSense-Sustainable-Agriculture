#!/usr/bin/env python3
"""
Comprehensive Project File Organizer
Moves all files from root into organized directory structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class ProjectOrganizer:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.moved_files = []
        self.skipped_files = []
        self.errors = []
        
        # Define file mappings: pattern -> destination
        self.file_mappings = {
            # Backend code
            'chatbot_': 'src/backend/ai/',
            'crop_': 'src/backend/ai/',
            'disease_': 'src/backend/ai/',
            'weed_': 'src/backend/ai/',
            'weather': 'src/backend/integrations/',
            'llm_': 'src/backend/integrations/',
            'ml_': 'src/backend/ai/',
            'phi_': 'src/backend/ai/',
            'vlm_': 'src/backend/ai/',
            'rag_': 'src/backend/ai/',
            'hybrid_': 'src/backend/ai/',
            'scold_': 'src/backend/ai/',
            'plant_health': 'src/backend/ai/',
            'comprehensive_disease': 'src/backend/ai/',
            'enhanced_weed': 'src/backend/ai/',
            'smart_farming': 'src/backend/ai/',
            'smart_weed': 'src/backend/ai/',
            'yield_': 'src/backend/ai/',
            'nlp_': 'src/backend/nlp/',
            
            # Database & Configuration
            'database_': 'src/backend/database/',
            'auth_': 'src/backend/auth/',
            'config': 'config/',
            'models.py': 'src/backend/models/',
            'core': 'src/backend/core/',
            'middleware': 'src/backend/middleware/',
            'metrics': 'src/backend/monitoring/',
            'rate_limiter': 'src/backend/security/',
            'notifier': 'src/backend/notifications/',
            
            # IoT related
            '.ino': 'src/iot/arduino/',
            '.cpp': 'src/iot/',
            
            # Data & Datasets
            '.csv': 'data/datasets/',
            'india_crop': 'data/datasets/',
            'Crop_recommendation': 'data/datasets/',
            'synthetic_train': 'data/training-data/',
            
            # Models
            '.joblib': 'models/trained/',
            '.pkl': 'models/trained/',
            '.h5': 'models/trained/',
            '.pb': 'models/pretrained/',
            '.onnx': 'models/trained/',
            '.pt': 'models/trained/',
            '.bin': 'models/pretrained/',
            
            # Tests
            'test_': 'tests/unit/',
            'conftest.py': 'tests/',
            'e2e_': 'tests/e2e/',
            'locustfile': 'tests/performance/',
            'pytest.ini': 'tests/',
            'playwright': 'tests/e2e/',
            
            # Training & Scripts
            'retrain_': 'scripts/training/',
            'train_': 'scripts/training/',
            'start_': 'scripts/deployment/',
            'start.sh': 'scripts/deployment/',
            'check_': 'scripts/utilities/',
            'validate_': 'scripts/utilities/',
            'monitor_': 'scripts/monitoring/',
            'apply_': 'scripts/utilities/',
            'fix_': 'scripts/utilities/',
            'dev_launcher': 'scripts/utilities/',
            'setup_': 'scripts/setup/',
            'install_': 'scripts/setup/',
            'cleanup_': 'scripts/utilities/',
            'comprehensive_': 'tools/development/',
            'e2e_test': 'tests/e2e/',
            'e2e_local': 'tests/e2e/',
            
            # Documentation
            '.md': 'docs/guides/',
            'README': 'docs/',
            'ARCHITECTURE': 'docs/architecture/',
            'ENV_VARS': 'docs/setup/',
            'CUDA': 'docs/setup/',
            'NPU': 'docs/setup/',
            'WSL2': 'docs/setup/',
            'SCOLD': 'docs/guides/',
            'GENAI': 'docs/guides/',
            'CHATBOT': 'docs/guides/',
            'HARDWARE': 'docs/guides/',
            'PYTHON_312': 'docs/guides/',
            'ML_': 'docs/guides/',
            'GPU_': 'docs/guides/',
            'DEPLOYMENT': 'docs/deployment/',
            'PRODUCTION': 'docs/deployment/',
            'E2E_': 'docs/guides/',
            'QUICKSTART': 'docs/setup/',
            'TESTING': 'docs/guides/',
            'TROUBLESHOOTING': 'docs/troubleshooting/',
            'openapi.json': 'docs/api-reference/',
            
            # Reports
            'analysis_report': 'reports/analysis/',
            'CLEANUP_LOG': 'reports/cleanup/',
            'E2E_ANALYSIS': 'reports/analysis/',
            'E2E_CLEANUP': 'reports/cleanup/',
            'ML_MODEL_TEST': 'reports/performance/',
            'npu_benchmark': 'reports/benchmarks/',
            'retraining_report': 'reports/performance/',
            
            # Examples
            'examples_': 'examples/integration/',
            'examples/': 'examples/',
            
            # Frontend
            'farm-fortune': 'src/frontend/',
            'frontend_': 'src/frontend/',
            '.html': 'src/frontend/',
            
            # Tools
            'security_audit': 'tools/security/',
            'generate_blueprint': 'tools/development/',
            'demo_': 'tools/development/',
            'e2e_local_runner': 'tools/development/',
            
            # Configuration & Docker
            '.env': 'config/environments/',
            'docker': 'config/docker/',
            'docker-compose': 'config/docker/',
            'Dockerfile': 'config/docker/',
            'tsconfig.json': 'config/',
            '.gitignore': 'config/',
            
            # Notebooks
            '.ipynb': 'notebooks/',
        }
    
    def organize(self, dry_run=False):
        """Organize all files"""
        print(f"{'[DRY RUN]' if dry_run else '[EXECUTING]'} Starting file organization...")
        
        for file_path in self.root.glob('*'):
            if file_path.is_dir():
                # Skip directories we want to keep in root or already organized
                if file_path.name in ['src', 'data', 'models', 'tests', 'docs', 'scripts', 
                                     'config', 'reports', 'tools', 'examples', '.git', 
                                     '__pycache__', 'notebooks', '.vscode', '.github',
                                     'agrisense_app', 'AGRISENSE_IoT', 'agrisense_pi_edge_minimal',
                                     'aiml_backend_from_docker', 'AI_Models', 'cleanup',
                                     'node_modules', '.pytest_cache', 'documentation', 'docker',
                                     'docs', 'e2e', '.venv', 'venv']:
                    continue
                    
                # Move unorganized backend directories
                if file_path.name == 'agrisense_app':
                    self._move_file(file_path, 'src/backend/agrisense_app', dry_run)
                elif file_path.name in ['AGRISENSE_IoT', 'agrisense_pi_edge_minimal']:
                    self._move_file(file_path, 'src/iot/', dry_run)
                elif file_path.name == 'AI_Models':
                    self._move_file(file_path, 'models/', dry_run)
                elif file_path.name in ['datasets', 'training_data']:
                    self._move_file(file_path, 'data/', dry_run)
                continue
            
            # Process files
            dest = self._find_destination(file_path.name)
            if dest:
                self._move_file(file_path, dest, dry_run)
            else:
                # Keep root-level important files
                if file_path.name not in ['README.md', 'LICENSE', '.gitignore', '.env.example',
                                         'package.json', 'package-lock.json', 'requirements.txt']:
                    self.skipped_files.append(str(file_path.name))
    
    def _find_destination(self, filename):
        """Find appropriate destination for a file"""
        # Check by exact patterns first
        for pattern, dest in self.file_mappings.items():
            if pattern in filename or filename.endswith(pattern):
                return dest
        return None
    
    def _move_file(self, source, dest_dir, dry_run=False):
        """Move a file to destination directory"""
        try:
            dest_path = self.root / dest_dir
            dest_path.mkdir(parents=True, exist_ok=True)
            
            dest_file = dest_path / source.name
            
            if dry_run:
                print(f"  [WOULD MOVE] {source.name} → {dest_dir}")
                self.moved_files.append((str(source), str(dest_file)))
            else:
                if dest_file.exists():
                    print(f"  [SKIP] {source.name} (already exists in {dest_dir})")
                    self.skipped_files.append(str(source.name))
                else:
                    shutil.move(str(source), str(dest_file))
                    print(f"  [MOVED] {source.name} → {dest_dir}")
                    self.moved_files.append((str(source), str(dest_file)))
        except Exception as e:
            print(f"  [ERROR] Failed to move {source.name}: {e}")
            self.errors.append(f"{source.name}: {e}")
    
    def print_summary(self):
        """Print organization summary"""
        print("\n" + "="*60)
        print("ORGANIZATION SUMMARY")
        print("="*60)
        print(f"Files moved: {len(self.moved_files)}")
        print(f"Files skipped: {len(self.skipped_files)}")
        print(f"Errors: {len(self.errors)}")
        
        if self.skipped_files:
            print(f"\nSkipped files ({len(self.skipped_files)}):")
            for f in self.skipped_files[:10]:
                print(f"  - {f}")
            if len(self.skipped_files) > 10:
                print(f"  ... and {len(self.skipped_files) - 10} more")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AgriSense File Organizer')
    parser.add_argument('--execute', action='store_true', help='Execute organization (default is dry-run)')
    parser.add_argument('--root', default=r"f:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK\AgriSense",
                       help='Project root directory')
    
    args = parser.parse_args()
    
    if not args.execute:
        print("="*60)
        print("DRY RUN MODE - No files will be moved")
        print("="*60)
        print("To execute, run: python organize_files.py --execute")
        print()
    
    organizer = ProjectOrganizer(args.root)
    organizer.organize(dry_run=not args.execute)
    organizer.print_summary()

if __name__ == "__main__":
    main()
