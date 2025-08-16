# Jupbox - Next.js + Electron + Spotify Integration

This is the modernized version of Jupbox that integrates Next.js with Electron for a better development experience and modern UI.

## 🚀 Features

- **Next.js Frontend**: Modern React-based UI with TypeScript and Tailwind CSS
- **Electron Integration**: Desktop app with native system access
- **Spotify API**: Full music playback control and search
- **Gesture Control**: Camera-based hand gesture detection for music control
- **Real-time Updates**: WebSocket connection for live status updates
- **Responsive Design**: Beautiful, modern UI that works on all screen sizes

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Desktop**: Electron 28
- **Backend**: Python Flask (existing backend)
- **Real-time**: Socket.IO
- **Styling**: Tailwind CSS with custom components

## 📁 Project Structure

```
jupbox/
├── frontend/                 # Next.js application
│   ├── src/app/             # Next.js app directory
│   ├── public/              # Static assets
│   └── package.json         # Frontend dependencies
├── backend/                  # Python Flask backend (existing)
├── main_next.js             # New Electron main process
├── preload.js               # Electron preload script
├── package.json             # Main package.json
└── README_NEXTJS.md         # This file
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (for backend)
- Spotify account and API credentials

### Installation

1. **Install main dependencies:**
   ```bash
   npm install
   ```

2. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Set up backend (existing setup):**
   ```bash
   cd backend
   pip install -r requirements.txt
   # Configure Spotify API credentials
   cd ..
   ```

### Development

1. **Start the backend server:**
   ```bash
   cd backend
   python main.py
   ```

2. **In a new terminal, start the Next.js development server:**
   ```bash
   npm run frontend:dev
   ```

3. **In another terminal, start the Electron app:**
   ```bash
   npm run dev:next
   ```

### Production Build

1. **Build the frontend:**
   ```bash
   npm run frontend:build
   ```

2. **Build the Electron app:**
   ```bash
   npm run build:next
   ```

## 🎯 Available Scripts

### Main Package
- `npm start` - Start Electron with original HTML
- `npm run start:next` - Start Electron with Next.js (development)
- `npm run dev:next` - Start Electron with Next.js (development)
- `npm run build:next` - Build frontend and Electron app
- `npm run frontend:dev` - Start Next.js development server
- `npm run frontend:build` - Build Next.js for production
- `npm run frontend:start` - Start Next.js production server

### Frontend Package
- `npm run dev` - Start Next.js development server
- `npm run build` - Build Next.js app
- `npm run build:static` - Build static export for Electron
- `npm start` - Start production server

## 🔧 Configuration

### Environment Variables

The app automatically detects whether it's running in Electron or standalone:

- **Electron Mode**: Uses IPC for system info, loads from localhost:3000 in dev
- **Standalone Mode**: Falls back to browser APIs, loads from localhost:3000

### Backend Configuration

Ensure your backend is running on `http://localhost:5000` with the following endpoints:
- `/api/spotify/user` - User authentication
- `/api/spotify/current` - Current track info
- `/api/spotify/play` - Play/pause control
- `/api/spotify/next` - Next track
- `/api/spotify/previous` - Previous track
- `/api/spotify/search` - Track search
- `/api/spotify/play-track` - Play specific track

## 🌟 Key Improvements

1. **Modern UI**: Beautiful, responsive design with Tailwind CSS
2. **Type Safety**: Full TypeScript support for better development experience
3. **Component Architecture**: Modular React components for maintainability
4. **Hot Reload**: Fast development with Next.js hot reload
5. **Better State Management**: React hooks for cleaner state handling
6. **Improved UX**: Better notifications, loading states, and error handling

## 🔄 Migration from Original

The original `index.html` and `renderer.js` files are preserved. To switch back:

1. Change `main` in `package.json` from `main_next.js` to `main.js`
2. Use `npm start` instead of `npm run dev:next`

## 🐛 Troubleshooting

### Common Issues

1. **Port 3000 already in use**: Kill the process or change the port in Next.js config
2. **Backend connection failed**: Ensure the Python backend is running on port 5000
3. **Spotify authentication failed**: Check your Spotify API credentials
4. **Electron won't start**: Verify Node.js version and dependencies

### Debug Mode

- **Frontend**: Use browser DevTools when running standalone
- **Electron**: DevTools open automatically in development mode
- **Backend**: Check Python console for errors

## 📝 Development Notes

- The app automatically detects Electron environment
- Socket.IO connection is established automatically
- All Spotify API calls use the same backend endpoints
- Gesture control works the same as the original version
- System info is enhanced when running in Electron

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see the main LICENSE file for details.

---

**Note**: This is the Next.js integrated version. The original vanilla JS version is still available in `main.js` and `renderer.js`. 