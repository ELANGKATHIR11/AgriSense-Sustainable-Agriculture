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
      const ollamaPath = 'C:\\Users\\elang\\AppData\\Local\\Programs\\Ollama\\ollama.exe';
      const fs = require('fs');
      if (fs.existsSync(ollamaPath)) {
        exec(`start /B "" "${ollamaPath}" serve`, (err) => {
          if (err) console.error('Failed to automatically boot Ollama:', err);
        });
      } else {
        exec('start /B ollama serve', (err) => {
          if (err) console.error('Failed to automatically boot Ollama:', err);
        });
      }
    } else {
      console.log('Ollama service detected on port 11434.');
    }
  });
}

function checkAndStartPostgres() {
  checkPort(5432, (inUse) => {
    if (!inUse) {
      console.log('PostgreSQL is offline. Attempting to start service...');
      const pgctlPath = 'F:\\Program Files\\PostgreSQL\\18\\bin\\pg_ctl.exe';
      const pgData = 'F:\\Program Files\\PostgreSQL\\18\\data';
      const fs = require('fs');
      if (fs.existsSync(pgctlPath)) {
        exec(`"${pgctlPath}" start -D "${pgData}"`, (err, stdout, stderr) => {
          if (err) {
            console.error('Failed to boot PostgreSQL via pg_ctl:', err);
            exec('net start postgresql-x64-18', (err2) => {
              if (err2) console.error('Failed to start postgresql service:', err2);
            });
          } else {
            console.log('PostgreSQL daemon started successfully via pg_ctl.');
          }
        });
      } else {
        exec('net start postgresql-x64-18', (err) => {
          if (err) console.error('Failed to start postgresql service:', err);
        });
      }
    } else {
      console.log('PostgreSQL service detected on port 5432.');
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
    // Development mode: Run start_backend.bat in a new window to prevent segfaults
    const batPath = path.join(__dirname, 'start_backend.bat');
    console.log(`Spawning dev backend via BAT in new window: ${batPath}`);
    backendProcess = spawn('cmd.exe', ['/c', 'start', 'cmd.exe', '/c', batPath], {
      cwd: __dirname,
      shell: false
    });
  }

  const fs = require('fs');
  const logStream = fs.createWriteStream(path.join(__dirname, 'backend_boot.log'), { flags: 'a' });

  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend stdout: ${data}`);
    logStream.write(`[${new Date().toISOString()}] STDOUT: ${data}\n`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`Backend stderr: ${data}`);
    logStream.write(`[${new Date().toISOString()}] STDERR: ${data}\n`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    logStream.write(`[${new Date().toISOString()}] Backend process exited with code ${code}\n`);
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

  // Open Developer Tools
  mainWindow.webContents.openDevTools();

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
  checkAndStartPostgres();
  // Wait a short moment for database port to bind before spawning backend
  setTimeout(() => {
    startBackend();
    createWindow();
  }, 1500);
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
