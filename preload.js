const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // System information
  getSystemInfo: () => ipcRenderer.invoke('get-system-info'),
  
  // Spotify API base URL
  spotifyApiBase: 'http://localhost:5000/api/spotify',
  
  // WebSocket URL
  webSocketUrl: 'http://localhost:5000',
  
  // Platform detection
  platform: process.platform,
  
  // App version
  appVersion: process.env.npm_package_version || '1.0.0'
});

// Expose a global flag to indicate we're running in Electron
contextBridge.exposeInMainWorld('isElectron', true); 