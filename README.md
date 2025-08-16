# 🎵 Jupbox - Integrated Music Control with Gesture Detection

A real-time music control system that combines Spotify integration with hand gesture detection using your webcam. Control your music playback by showing hand gestures to the camera!

## ✨ Features

- **Spotify Integration**: Full Spotify API integration with playback control
- **Gesture Detection**: Real-time hand gesture recognition using Roboflow API
- **Web Interface**: Beautiful Electron-based frontend with real-time updates
- **WebSocket Communication**: Real-time communication between camera and frontend
- **🎮 Gesture Controls**: 
  - Show **fist** (zero) to **PAUSE** music
  - Show **open palm** (five) to **PLAY** music

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Node.js (for Electron frontend)
- Webcam
- Spotify account

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jupbox
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Integrated System

#### Option 1: Using the startup script (Recommended)
```bash
python start_integrated.py
```

#### Option 2: Manual startup
```bash
# Terminal 1: Start the integrated backend
cd backend
python main.py

# Terminal 2: Start the Electron frontend (optional)
cd frontend
npm start
```

## 🎮 How to Use

1. **Start the system** using one of the methods above
2. **Open your webcam** - the camera window will appear automatically
3. **Show hand gestures** to control music:
   - **Show fist** → **PAUSE** the current track
   - **Show palm** → **PLAY** the current track
4. **Use the web interface** to:
   - View current track information
   - Search for new tracks
   - Control playback manually
   - Monitor camera connection status

## 🏗️ Architecture

### Backend (`backend/main.py`)
- **Flask Server**: REST API for Spotify control
- **WebSocket Server**: Real-time communication
- **Camera Detection**: Hand gesture recognition using Roboflow API
- **Spotify Integration**: Full playback control via Spotify Web API

### Frontend (`frontend/`)
- **Electron App**: Desktop application interface
- **WebSocket Client**: Real-time updates from camera
- **Spotify Player**: Music playback interface
- **Camera Status**: Real-time connection monitoring

## 🔧 Configuration

### Spotify Setup
1. Create a Spotify app at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Get your `CLIENT_ID` and `CLIENT_SECRET`
3. Update the credentials in `backend/main.py`:
   ```python
   SPOTIFY_CLIENT_ID = "your_client_id"
   SPOTIFY_CLIENT_SECRET = "your_client_secret"
   ```

## 📁 Project Structure

```
jupbox/
├── backend/
│   ├── main.py              # Integrated backend (camera + Flask + WebSocket)
│   ├── spotify_api.py       # Spotify API endpoints
│   ├── spotify_auth.py      # Spotify authentication
│   ├── requirements.txt     # Python dependencies
│   └── webcam_raw.py        # Original camera-only version
├── frontend/
│   ├── index.html           # Main interface
│   ├── renderer.js          # Frontend logic with WebSocket
│   ├── styles.css           # Styling
│   ├── package.json         # Node.js dependencies
│   └── main.js              # Electron main process
├── start_integrated.py      # Startup script
└── README.md               # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**
   - Check if webcam is available
   - Ensure camera permissions are granted
   - Try restarting the application

2. **Spotify connection failed**
   - Verify Spotify credentials in `backend/main.py`
   - Check if Spotify is running and authorized
   - Ensure internet connection is stable

3. **WebSocket connection failed**
   - Check if backend is running on port 5000
   - Ensure no firewall is blocking the connection
   - Try refreshing the frontend

4. **Gesture detection not working**
   - Ensure good lighting conditions
   - Hold hand gestures clearly in front of camera
   - Check Roboflow API key and usage limits

## Stack

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- [Roboflow](https://roboflow.com/) for gesture detection
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Socket.IO](https://socket.io/) for real-time communication
- [Electron](https://www.electronjs.org/) for the desktop app framework

---

:)
