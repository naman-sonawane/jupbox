const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false
    },
    icon: path.join(__dirname, 'assets/icon.png') 
  });

  
  mainWindow.loadFile('index.html');

  
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  
  mainWindow.on('closed', () => {
    
    mainWindow = null;
  });
}


app.whenReady().then(createWindow);


app.on('window-all-closed', () => {
  
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  
  
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
