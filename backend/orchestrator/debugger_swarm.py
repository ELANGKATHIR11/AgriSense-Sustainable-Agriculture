"""
AGRISENSE Debugger Swarm - Self-Healing & Automated Debugging Engine
Analyzes stack traces, parses error locations, applies patches, runs unit tests, and supports rollbacks.
"""

import os
import re
import subprocess
import shutil
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("DebuggerSwarm")

class DebuggerSwarm:
    def __init__(self, project_root: str = "."):
        self.project_root = os.path.abspath(project_root)
        self.backups = {}

    def parse_stack_trace(self, stack_trace: str) -> Tuple[str, int, str]:
        """
        Parses a Python traceback to find the file, line number, and error message.
        """
        # Search for file and line number patterns in standard tracebacks
        # E.g. File "backend/main.py", line 45, in some_function
        matches = re.findall(r'File "([^"]+)", line (\d+)', stack_trace)
        
        # Get the last matching file inside our project directory
        target_file = ""
        line_num = -1
        
        if matches:
            # Iterate backwards to find the leaf error in user code (not library code)
            for file_path, line in reversed(matches):
                # Check if file path is in the workspace
                full_path = os.path.join(self.project_root, file_path)
                if os.path.exists(full_path) and "site-packages" not in full_path and "lib/python" not in full_path:
                    target_file = full_path
                    line_num = int(line)
                    break
            
            if not target_file:
                # Fallback to the absolute last entry if none matched the local project filter
                file_path, line = matches[-1]
                target_file = os.path.abspath(file_path)
                line_num = int(line)
        
        # Find the last line of the traceback which contains the error message
        lines = [line.strip() for line in stack_trace.split("\n") if line.strip()]
        error_msg = lines[-1] if lines else "Unknown Error"
        
        return target_file, line_num, error_msg

    def backup_file(self, file_path: str):
        """Creates a temporary backup of the target file to allow rollbacks."""
        if not os.path.exists(file_path):
            return
        
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        self.backups[file_path] = backup_path
        logger.info(f"Created backup of {file_path} at {backup_path}")

    def rollback_file(self, file_path: str):
        """Restores file from backup and deletes backup file."""
        backup_path = self.backups.get(file_path)
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            os.remove(backup_path)
            del self.backups[file_path]
            logger.info(f"Rolled back {file_path} from backup.")
        else:
            logger.warning(f"No backup found for {file_path}.")

    def generate_patch(self, file_path: str, line_num: int, error_msg: str) -> str:
        """
        Generates a patch using standard engineering heuristics (e.g. adding null guards,
        type casts, try/except blocks) to heal common runtime errors.
        """
        if not os.path.exists(file_path):
            return ""

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_num <= 0 or line_num > len(lines):
            return ""

        target_line = lines[line_num - 1]
        indent = len(target_line) - len(target_line.lstrip())
        indent_str = target_line[:indent]

        patch_description = f"Healed: {error_msg} at line {line_num}"

        # Heuristics based on error messages
        if "NoneType" in error_msg or "'None'" in error_msg or "AttributeError" in error_msg:
            # Null guard heuristic: wrap line in a None check
            # E.g. val.method() -> if val is not None: val.method()
            stripped_line = target_line.strip()
            # Find the object being called or accessed
            match = re.match(r"([a-zA-Z_][a-zA-Z0-9_.]*)", stripped_line)
            if match:
                obj_name = match.group(1).split(".")[0]
                lines[line_num - 1] = f"{indent_str}if {obj_name} is not None:\n{indent_str}    {stripped_line}\n"
        elif "KeyError" in error_msg:
            # KeyError guard: use .get() instead of brackets or add condition
            stripped_line = target_line.strip()
            match = re.findall(r"\[['\"]([^'\"]+)['\"]\]", stripped_line)
            if match:
                key = match[0]
                # Replace dict['key'] with dict.get('key')
                fixed_line = re.sub(rf"\[['\"]{key}['\"]\]", f".get('{key}')", stripped_line)
                lines[line_num - 1] = f"{indent_str}{fixed_line}\n"
        elif "ZeroDivisionError" in error_msg:
            # Add divide-by-zero protection
            stripped_line = target_line.strip()
            # Locate divisor
            parts = stripped_line.split("/")
            if len(parts) > 1:
                divisor = parts[-1].strip()
                lines[line_num - 1] = f"{indent_str}if {divisor} != 0:\n{indent_str}    {stripped_line}\n"
        else:
            # Default fallback: wrap in a general try/except
            stripped_line = target_line.strip()
            lines[line_num - 1] = f"{indent_str}try:\n{indent_str}    {stripped_line}\n{indent_str}except Exception as e:\n{indent_str}    logger.warning(f'Self-healed exception: {{e}}')\n"

        # Write patch back to file
        self.backup_file(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return patch_description

    def run_verification_tests(self) -> bool:
        """Runs pytest suite to verify project health after patch application."""
        try:
            res = subprocess.run(
                ["pytest", "backend/tests/"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return res.returncode == 0
        except Exception as e:
            logger.error(f"Failed to execute verification tests: {e}")
            return False

    def heal_traceback(self, stack_trace: str) -> Dict[str, Any]:
        """
        Main entry point for self-healing swarms.
        Parses stack trace, backs up files, applies patches, tests changes, and rolls back if tests fail.
        """
        file_path, line_num, error_msg = self.parse_stack_trace(stack_trace)
        if not file_path or line_num <= 0:
            return {
                "success": False,
                "action": "none",
                "error": "Traceback did not match any workspace file."
            }

        logger.info(f"Attempting self-healing on {file_path} at line {line_num} due to: {error_msg}")
        
        try:
            patch_desc = self.generate_patch(file_path, line_num, error_msg)
            if not patch_desc:
                return {
                    "success": False,
                    "action": "none",
                    "error": "Failed to generate valid patch template."
                }

            # Run test validation
            logger.info("Running verification test suite...")
            tests_pass = self.run_verification_tests()
            
            if tests_pass:
                logger.info("Verification tests passed. Patch successfully applied and committed.")
                # Clean up backup
                if file_path in self.backups:
                    os.remove(self.backups[file_path])
                    del self.backups[file_path]
                return {
                    "success": True,
                    "action": "patch_applied",
                    "file": file_path,
                    "line": line_num,
                    "description": patch_desc
                }
            else:
                logger.warning("Verification tests failed after patching. Rolling back changes.")
                self.rollback_file(file_path)
                return {
                    "success": False,
                    "action": "rollback",
                    "file": file_path,
                    "line": line_num,
                    "error": "Verification tests failed post-patch."
                }
        except Exception as e:
            logger.error(f"Self-healing execution failed: {e}")
            self.rollback_file(file_path)
            return {
                "success": False,
                "action": "rollback",
                "file": file_path,
                "line": line_num,
                "error": f"Exception raised during healing: {str(e)}"
            }

debugger_swarm = DebuggerSwarm()
