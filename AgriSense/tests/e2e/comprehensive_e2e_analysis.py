#!/usr/bin/env python3
"""
Comprehensive E2E Project Analysis Script
Analyzes entire AgriSense project and generates cleanup recommendations
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class ProjectAnalyzer:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.analysis = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(project_root),
            'files_by_type': defaultdict(int),
            'duplicate_scripts': [],
            'obsolete_files': [],
            'unused_dependencies': [],
            'documentation_files': [],
            'test_files': [],
            'cleanup_candidates': [],
            'directory_stats': {},
            'total_size_mb': 0
        }
        
    def analyze(self):
        """Run complete project analysis"""
        print("[*] Starting comprehensive project analysis...")
        
        self._scan_files()
        self._find_duplicates()
        self._identify_obsolete_files()
        self._analyze_dependencies()
        self._categorize_docs()
        self._check_venvs()
        
        return self.analysis
    
    def _scan_files(self):
        """Scan all files and categorize them"""
        print("[*] Scanning all files...")
        
        try:
            for root, dirs, files in os.walk(self.root):
                # Skip virtual environments and node_modules
                dirs[:] = [d for d in dirs if d not in ['.venv', 'venv312', 'venv_ml312', 'venv_npu', '.venv312', '.venv.bak', 'node_modules', '__pycache__', '.pytest_cache']]
                
                rel_path = Path(root).relative_to(self.root)
                
                for file in files:
                    file_path = Path(root) / file
                    ext = file_path.suffix.lower()
                    
                    self.analysis['files_by_type'][ext] += 1
                    
                    try:
                        size_kb = file_path.stat().st_size / 1024
                        self.analysis['total_size_mb'] += size_kb / 1024
                    except:
                        pass
        except Exception as e:
            print(f"[!] Error scanning files: {e}")
    
    def _find_duplicates(self):
        """Find duplicate or redundant scripts"""
        print("[*] Finding duplicate scripts...")
        
        # Known duplicate patterns
        duplicates = {
            'start': ['start_agrisense.bat', 'start_agrisense.ps1', 'start_agrisense.py', 'start.sh', 'start_agrisense_scold.ps1', 'start_optimized.ps1', 'start_hybrid_ai.ps1'],
            'train': ['retrain_all_models_gpu.py', 'retrain_fast_gpu.py', 'retrain_fast_gpu.sh', 'retrain_gpu.sh', 'retrain_gpu_simple.py', 'retrain_gpu_simple.sh', 'retrain_production.py', 'retrain_production.sh', 'train_npu_models.bat', 'train_npu_models.ps1'],
            'cleanup': ['cleanup_optimize_project.ps1', 'comprehensive_cleanup.ps1'],
            'test': ['test_gpu_backend.sh', 'test_integration.ps1', 'test_frontend_api_integration.ps1', 'test_human_chatbot.py', 'test_hybrid_ai.py', 'test_ml_endpoints.py', 'test_ml_models_comprehensive.py', 'test_phi_chatbot.py', 'test_scold_integration.py'],
            'install': ['install_cuda_wsl2.bat', 'install_cuda_wsl2.ps1', 'install_cuda_wsl2.sh']
        }
        
        for category, files in duplicates.items():
            if len(files) > 1:
                self.analysis['duplicate_scripts'].append({
                    'category': category,
                    'files': files,
                    'recommendation': f'Keep only best-performing version, remove {len(files)-1} variants'
                })
    
    def _identify_obsolete_files(self):
        """Identify obsolete and redundant files"""
        print("[*] Identifying obsolete files...")
        
        obsolete_patterns = [
            'cleanup_backup*',  # Backup directories
            '*_report*.md',     # Numerous duplicate reports
            '*.log',            # Log files
            'tmp*.py',          # Temporary test files
            'test_*.py',        # Duplicate test files
            '*CLEANUP*.md',     # Cleanup reports
            '*OPTIMIZATION*.md', # Optimization reports
            '*EVALUATION*.md',   # Evaluation reports
            'temp_*.onnx*',     # Temporary models
            '.file_sizes.json', # Metadata files
            '.sizes_summary.json',
            '.pip_freeze.txt',
        ]
        
        root = Path(self.root)
        
        for pattern in obsolete_patterns:
            for file_path in root.rglob(pattern):
                if file_path.is_file():
                    try:
                        size_mb = file_path.stat().st_size / (1024*1024)
                        self.analysis['obsolete_files'].append({
                            'path': str(file_path.relative_to(root)),
                            'size_mb': round(size_mb, 2),
                            'pattern': pattern
                        })
                    except:
                        pass
    
    def _analyze_dependencies(self):
        """Analyze dependencies in requirements files"""
        print("[*] Analyzing dependencies...")
        
        req_files = list(self.root.rglob('requirements*.txt'))
        
        all_deps = {}
        for req_file in req_files:
            try:
                with open(req_file, 'r', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            pkg = re.split(r'[<>=!\[\]]', line)[0].strip()
                            if pkg and pkg not in ['', '-']:
                                all_deps[pkg] = req_file.name
            except:
                pass
        
        # Common unused patterns
        unused_patterns = ['tensorflow', 'torch', 'cuda', 'scold', 'phi']
        
        for pattern in unused_patterns:
            for dep in list(all_deps.keys()):
                if pattern.lower() in dep.lower():
                    self.analysis['unused_dependencies'].append({
                        'package': dep,
                        'source': all_deps[dep],
                        'status': 'possibly_unused'
                    })
    
    def _categorize_docs(self):
        """Categorize documentation files"""
        print("[*] Categorizing documentation...")
        
        docs = list(self.root.glob('*.md')) + list(self.root.glob('docs/*.md'))
        
        for doc in docs:
            try:
                self.analysis['documentation_files'].append({
                    'name': doc.name,
                    'size_kb': doc.stat().st_size / 1024
                })
            except:
                pass
    
    def _check_venvs(self):
        """Check for multiple virtual environments"""
        print("[*] Checking virtual environments...")
        
        venv_dirs = ['venv', 'venv312', 'venv_ml312', 'venv_npu', '.venv', '.venv312', '.venv.bak']
        
        for venv_name in venv_dirs:
            venv_path = self.root / venv_name
            if venv_path.exists():
                try:
                    total_size = sum(f.stat().st_size for f in venv_path.rglob('*') if f.is_file())
                    self.analysis['cleanup_candidates'].append({
                        'type': 'venv',
                        'name': venv_name,
                        'path': str(venv_path.relative_to(self.root)),
                        'size_mb': round(total_size / (1024*1024), 2),
                        'recommendation': 'Keep only active venv, remove unused ones'
                    })
                except:
                    pass

def main():
    project_root = r"f:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK\AgriSense"
    
    analyzer = ProjectAnalyzer(project_root)
    analysis = analyzer.analyze()
    
    # Save analysis
    report_file = Path(project_root) / "E2E_ANALYSIS_REPORT.json"
    with open(report_file, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json.dump({k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in analysis.items()}, 
                 f, indent=2, default=str)
    
    print(f"\n[+] Analysis complete! Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROJECT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total project size: {analysis['total_size_mb']:.2f} MB")
    print(f"Files by extension: {dict(analysis['files_by_type'])}")
    print(f"\nDuplicate script groups: {len(analysis['duplicate_scripts'])}")
    print(f"Obsolete files found: {len(analysis['obsolete_files'])}")
    print(f"Possibly unused dependencies: {len(analysis['unused_dependencies'])}")
    print(f"Cleanup candidates (venvs, etc): {len(analysis['cleanup_candidates'])}")
    
    # Print cleanup candidates
    if analysis['cleanup_candidates']:
        print("\n[!] MAJOR CLEANUP OPPORTUNITIES:")
        total_venv_size = 0
        for item in analysis['cleanup_candidates']:
            if item['type'] == 'venv':
                print(f"  - {item['name']}: {item['size_mb']} MB")
                total_venv_size += item['size_mb']
        print(f"\n  Total venv size that can be recovered: {total_venv_size:.2f} MB")

if __name__ == "__main__":
    main()
