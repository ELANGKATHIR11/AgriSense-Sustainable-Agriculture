const path = require('path');
const fs = require('fs');
const os = require('os');

/**
 * Get the Python executable path
 * Prioritizes virtual environment, then system Python
 */
function getPythonPath() {
  // Try virtual environment first (relative to backend directory)
  const venvPaths = [
    path.join(__dirname, '../../.venv/Scripts/python.exe'), // Windows
    path.join(__dirname, '../../.venv/bin/python'), // Linux/Mac
    path.join(__dirname, '../.venv/Scripts/python.exe'), // Alternative Windows
    path.join(__dirname, '../.venv/bin/python') // Alternative Linux/Mac
  ];

  for (const pythonPath of venvPaths) {
    if (fs.existsSync(pythonPath)) {
      console.log(`✅ Using Python from virtual environment: ${pythonPath}`);
      return pythonPath;
    }
  }

  // Fallback to system Python
  const systemPython = os.platform() === 'win32' ? 'python.exe' : 'python3';
  console.warn(`⚠️  Virtual environment Python not found, using system: ${systemPython}`);
  return systemPython;
}

module.exports = {
  pythonPath: getPythonPath(),
  getPythonPath
};
