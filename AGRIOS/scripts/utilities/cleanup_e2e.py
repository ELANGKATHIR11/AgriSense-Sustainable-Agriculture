#!/usr/bin/env python3
"""
E2E Cleanup Execution Script
Safely removes unwanted files and directories from AgriSense project
Run with: python cleanup_e2e.py --execute
"""

import os
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime

class ProjectCleaner:
    def __init__(self, project_root, dry_run=True):
        self.root = Path(project_root)
        self.dry_run = dry_run
        self.cleanup_log = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'deleted_items': [],
            'errors': [],
            'summary': {}
        }
        
    def cleanup(self):
        """Execute cleanup operations"""
        print(f"{'[DRY RUN]' if self.dry_run else '[EXECUTING]'} Starting E2E cleanup...")
        
        self._cleanup_venvs()
        self._cleanup_temp_files()
        self._cleanup_logs()
        self._cleanup_reports()
        self._cleanup_duplicates()
        self._update_gitignore()
        
        self._save_log()
        self._print_summary()
    
    def _cleanup_venvs(self):
        """Remove virtual environments"""
        print("\n[*] Cleaning up virtual environments...")
        venv_dirs = ['venv312', 'venv_ml312', 'venv_npu', '.venv', '.venv312', '.venv.bak']
        
        for venv in venv_dirs:
            venv_path = self.root / venv
            if venv_path.exists():
                try:
                    size_mb = sum(f.stat().st_size for f in venv_path.rglob('*')) / (1024*1024)
                    print(f"  Removing {venv}: {size_mb:.2f} MB")
                    
                    if not self.dry_run:
                        shutil.rmtree(venv_path)
                    
                    self.cleanup_log['deleted_items'].append({
                        'type': 'venv',
                        'path': venv,
                        'size_mb': size_mb
                    })
                except Exception as e:
                    print(f"  [!] Error removing {venv}: {e}")
                    self.cleanup_log['errors'].append(f"Failed to remove {venv}: {e}")
    
    def _cleanup_temp_files(self):
        """Remove temporary files"""
        print("\n[*] Cleaning up temporary files...")
        
        temp_patterns = [
            'tmp_*.py',
            '*.log',
            'temp_*.onnx.data',
            '.file_sizes.json',
            '.sizes_summary.json',
            '.pip_freeze.txt',
        ]
        
        for pattern in temp_patterns:
            for file_path in self.root.glob(pattern):
                if file_path.is_file():
                    try:
                        size_kb = file_path.stat().st_size / 1024
                        print(f"  Removing {file_path.name}: {size_kb:.2f} KB")
                        
                        if not self.dry_run:
                            file_path.unlink()
                        
                        self.cleanup_log['deleted_items'].append({
                            'type': 'temp_file',
                            'path': file_path.name,
                            'size_kb': size_kb
                        })
                    except Exception as e:
                        self.cleanup_log['errors'].append(f"Failed to remove {file_path.name}: {e}")
    
    def _cleanup_logs(self):
        """Remove log files"""
        print("\n[*] Cleaning up log files...")
        
        for log_file in self.root.glob('*.log'):
            if log_file.is_file():
                try:
                    size_kb = log_file.stat().st_size / 1024
                    print(f"  Removing {log_file.name}: {size_kb:.2f} KB")
                    
                    if not self.dry_run:
                        log_file.unlink()
                    
                    self.cleanup_log['deleted_items'].append({
                        'type': 'log',
                        'path': log_file.name,
                        'size_kb': size_kb
                    })
                except Exception as e:
                    self.cleanup_log['errors'].append(f"Failed to remove {log_file.name}: {e}")
    
    def _cleanup_reports(self):
        """Remove obsolete report files"""
        print("\n[*] Cleaning up obsolete reports...")
        
        report_patterns = [
            'CLEANUP_*.md',
            'OPTIMIZATION_*.md',
            'ML_EVALUATION_*.md',
            '*_SUMMARY.md',
            'POST_CLEANUP_*.md',
            'CRITICAL_FIXES_*.md',
            'FINAL_VALIDATION_*.md',
            'COMPREHENSIVE_*.md',
            'DEPLOYMENT_CLEANUP_*.md',
            'PYTHON_312_OPTIMIZATION_*.md',
            'GPU_TRAINING_*.md',
            'NPU_TRAINING_*.md',
            'INTEGRATION_FIX*.md',
            'RETRAINING_COMPLETE*.md',
            'TROUBLESHOOTING_COMPLETE*.md',
            '*EVALUATION_FINAL*.txt',
        ]
        
        for pattern in report_patterns:
            for file_path in self.root.glob(pattern):
                if file_path.is_file():
                    try:
                        size_kb = file_path.stat().st_size / 1024
                        print(f"  Removing {file_path.name}: {size_kb:.2f} KB")
                        
                        if not self.dry_run:
                            file_path.unlink()
                        
                        self.cleanup_log['deleted_items'].append({
                            'type': 'report',
                            'path': file_path.name,
                            'size_kb': size_kb
                        })
                    except Exception as e:
                        self.cleanup_log['errors'].append(f"Failed to remove {file_path.name}: {e}")
        
        # Remove backup directories
        if (self.root / 'cleanup_backup_20251205_182237').exists():
            try:
                size_mb = sum(f.stat().st_size for f in (self.root / 'cleanup_backup_20251205_182237').rglob('*')) / (1024*1024)
                print(f"  Removing cleanup_backup_20251205_182237: {size_mb:.2f} MB")
                
                if not self.dry_run:
                    shutil.rmtree(self.root / 'cleanup_backup_20251205_182237')
                
                self.cleanup_log['deleted_items'].append({
                    'type': 'backup_dir',
                    'path': 'cleanup_backup_20251205_182237',
                    'size_mb': size_mb
                })
            except Exception as e:
                self.cleanup_log['errors'].append(f"Failed to remove cleanup_backup: {e}")
    
    def _cleanup_duplicates(self):
        """Remove duplicate script files"""
        print("\n[*] Removing duplicate startup scripts...")
        
        # Keep only essential versions, remove others
        startup_scripts_to_remove = [
            'start.sh',  # Will use Python scripts
            'start_agrisense_scold.ps1',
            'start_agrisense.bat',
            'start_hybrid_ai.ps1',
        ]
        
        for script in startup_scripts_to_remove:
            script_path = self.root / script
            if script_path.exists():
                try:
                    size_kb = script_path.stat().st_size / 1024
                    print(f"  Removing {script}: {size_kb:.2f} KB")
                    
                    if not self.dry_run:
                        script_path.unlink()
                    
                    self.cleanup_log['deleted_items'].append({
                        'type': 'duplicate_script',
                        'path': script,
                        'size_kb': size_kb
                    })
                except Exception as e:
                    self.cleanup_log['errors'].append(f"Failed to remove {script}: {e}")
        
        # Note: Keep one comprehensive training script pattern
        print("\n[*] Note: Training scripts (retrain_*.py) kept for backward compatibility")
        print("    Consider consolidating to scripts/train.py in future")
    
    def _update_gitignore(self):
        """Update .gitignore to prevent future venv commits"""
        print("\n[*] Updating .gitignore...")
        
        gitignore_path = self.root / '.gitignore'
        
        venv_entries = [
            '# Virtual Environments',
            'venv/',
            'venv312/',
            'venv_ml312/',
            'venv_npu/',
            '.venv/',
            '.venv312/',
            '.venv.bak/',
            '',
        ]
        
        if not self.dry_run:
            try:
                with open(gitignore_path, 'r') as f:
                    content = f.read()
                
                # Check if venv entries already exist
                if 'venv312/' not in content:
                    with open(gitignore_path, 'a') as f:
                        f.write('\n# Virtual Environments (Added ' + datetime.now().strftime('%Y-%m-%d') + ')\n')
                        f.write('\n'.join(venv_entries))
                    print("  Updated .gitignore with venv patterns")
                else:
                    print("  .gitignore already contains venv patterns")
            except Exception as e:
                self.cleanup_log['errors'].append(f"Failed to update .gitignore: {e}")
    
    def _save_log(self):
        """Save cleanup log"""
        log_file = self.root / f"CLEANUP_LOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate summary
        total_size_mb = sum(item.get('size_mb', item.get('size_kb', 0) / 1024) 
                           for item in self.cleanup_log['deleted_items'])
        
        self.cleanup_log['summary'] = {
            'total_items_deleted': len(self.cleanup_log['deleted_items']),
            'total_size_recovered_mb': round(total_size_mb, 2),
            'errors_count': len(self.cleanup_log['errors']),
            'status': 'dry_run' if self.dry_run else 'executed'
        }
        
        try:
            with open(log_file, 'w') as f:
                json.dump(self.cleanup_log, f, indent=2)
            print(f"\n[+] Cleanup log saved to: {log_file.name}")
        except Exception as e:
            print(f"[!] Failed to save cleanup log: {e}")
    
    def _print_summary(self):
        """Print cleanup summary"""
        summary = self.cleanup_log['summary']
        
        print("\n" + "="*60)
        print("CLEANUP SUMMARY")
        print("="*60)
        print(f"Status: {'DRY RUN' if self.dry_run else 'EXECUTED'}")
        print(f"Items processed: {summary['total_items_deleted']}")
        print(f"Space recovered: {summary['total_size_recovered_mb']:.2f} MB")
        print(f"Errors: {summary['errors_count']}")
        
        if self.cleanup_log['errors']:
            print("\n[!] Errors encountered:")
            for error in self.cleanup_log['errors']:
                print(f"  - {error}")
        
        if self.dry_run:
            print("\n[i] This was a DRY RUN. No files were actually deleted.")
            print("    To execute cleanup: python cleanup_e2e.py --execute")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AgriSense E2E Cleanup')
    parser.add_argument('--execute', action='store_true', help='Execute cleanup (default is dry-run)')
    parser.add_argument('--root', default=r"f:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK\AgriSense",
                       help='Project root directory')
    
    args = parser.parse_args()
    
    if not args.execute:
        print("="*60)
        print("DRY RUN MODE - No files will be deleted")
        print("="*60)
        print("To execute cleanup, run: python cleanup_e2e.py --execute")
        print()
    
    cleaner = ProjectCleaner(args.root, dry_run=not args.execute)
    cleaner.cleanup()

if __name__ == "__main__":
    main()
