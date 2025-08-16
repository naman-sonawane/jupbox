const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let nextProcess;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets/icon.png'), // Optional: add an icon
    show: false, // Don't show until ready
    titleBarStyle: 'default',
    autoHideMenuBar: false
  });

  // Load the Next.js app
  const isDev = process.env.NODE_ENV === 'development';
  
  if (isDev) {
    // In development, load from Next.js dev server
    mainWindow.loadURL('http://localhost:3000');
    
    // Open DevTools in development
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built Next.js app
    mainWindow.loadFile(path.join(__dirname, 'frontend/out/index.html'));
  }

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    require('electron').shell.openExternal(url);
    return { action: 'deny' };
  });
}

// Start Next.js development server
function startNextDev() {
  return new Promise((resolve, reject) => {
    console.log('🚀 Starting Next.js development server...');
    
    nextProcess = spawn('npm', ['run', 'dev'], {
      cwd: path.join(__dirname, 'frontend'),
      stdio: 'pipe',
      shell: true
    });

    nextProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Next.js:', output);
      
      // Check if Next.js is ready
      if (output.includes('Ready in') || output.includes('Local:')) {
        console.log('✅ Next.js development server is ready!');
        resolve();
      }
    });

    nextProcess.stderr.on('data', (data) => {
      console.error('Next.js Error:', data.toString());
    });

    nextProcess.on('error', (error) => {
      console.error('Failed to start Next.js:', error);
      reject(error);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      reject(new Error('Next.js server startup timeout'));
    }, 30000);
  });
}

// App event handlers
app.whenReady().then(async () => {
  try {
    if (process.env.NODE_ENV === 'development') {
      await startNextDev();
    }
    createWindow();
  } catch (error) {
    console.error('Failed to start app:', error);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on('before-quit', () => {
  // Clean up Next.js process
  if (nextProcess) {
    nextProcess.kill();
  }
});

// IPC handlers for communication between main and renderer
ipcMain.handle('get-system-info', () => {
  return {
    platform: process.platform,
    nodeVersion: process.version,
    electronVersion: process.versions.electron,
    appVersion: app.getVersion()
  };
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
}); 