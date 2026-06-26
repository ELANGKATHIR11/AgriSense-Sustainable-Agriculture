const { app, BrowserWindow } = require('electron');
const { spawn, exec } = require('child_process');
const path = require('path');
const http = require('http');

let backendProcess = null;
let mainWindow = null;
const PORT = 8000;

// Helper to check if a port is in use
function checkPort(port, callback) {
  const server = http.createServer();
  server.once('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      callback(true); // Port is active
    } else {
      callback(false);
    }
  });
  server.once('listening', () => {
    server.close();
    callback(false); // Port is free
  });
  server.listen(port, '127.0.0.1');
}

function checkAndStartOllama() {
  checkPort(11434, (inUse) => {
    if (!inUse) {
      console.log('Ollama is offline. Attempting to start service silently...');
      exec('start /B ollama serve', (err) => {
        if (err) console.error('Failed to automatically boot Ollama:', err);
      });
    } else {
      console.log('Ollama service detected on port 11434.');
    }
  });
}

function startBackend() {
  const isPackaged = app.isPackaged;
  
  if (isPackaged) {
    // Packaged mode: Execute the bundled PyInstaller binary
    const backendPath = path.join(process.resourcesPath, 'backend-dist', 'agrisense-backend', 'agrisense-backend.exe');
    console.log(`Spawning packaged backend: ${backendPath}`);
    backendProcess = spawn(backendPath, ['--host', '127.0.0.1', '--port', PORT.toString()]);
  } else {
    // Development mode: Run Python module
    const pythonExe = 'C:\\Users\\elang\\Miniconda3\\envs\\dgpu-core\\python.exe';
    console.log(`Spawning dev backend via Python: ${pythonExe}`);
    backendProcess = spawn(pythonExe, ['-m', 'backend.main'], {
      cwd: __dirname
    });
  }

  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend stdout: ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`Backend stderr: ${data}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: 'AgriSense Edge AI Dashboard',
    webPreferences: {
      nodeIntegration: false,
      contextBridge: true
    }
  });

  // Poll FastAPI gateway status on port 8000 and load the page once it responds 200
  const pollInterval = setInterval(() => {
    http.get(`http://127.0.0.1:${PORT}/api/health`, (res) => {
      if (res.statusCode === 200) {
        clearInterval(pollInterval);
        mainWindow.loadURL(`http://127.0.0.1:${PORT}`);
      }
    }).on('error', () => {
      // Backend still booting up, retry in 1s
    });
  }, 1000);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', () => {
  checkAndStartOllama();
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  if (backendProcess) {
    console.log('Terminating backend processes...');
    backendProcess.kill();
  }
});
