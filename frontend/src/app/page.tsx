'use client';

import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useSession, signOut } from 'next-auth/react';
import FaceLogin from '../components/FaceLogin';
import FaceEnrollment from '../components/FaceEnrollment';
import EmotionDetection from '../components/EmotionDetection';

// Use Electron API if available, otherwise fallback to localhost
const SPOTIFY_API_BASE = typeof window !== 'undefined' && (window as any).electronAPI 
  ? (window as any).electronAPI.spotifyApiBase 
  : 'http://localhost:5000/api/spotify';

const WEBSOCKET_URL = typeof window !== 'undefined' && (window as any).electronAPI 
  ? (window as any).electronAPI.webSocketUrl 
  : 'http://localhost:5000';

interface Track {
  name: string;
  artist: string;
  album: string;
  cover_url?: string;
  progress_ms: number;
  duration_ms: number;
  is_playing: boolean;
}

interface User {
  name: string;
  image?: string;
}

interface SystemInfo {
  platform: string;
  nodeVersion: string;
  electronVersion: string;
  appVersion: string;
}

export default function Home() {
  const { data: session, status } = useSession();
  const [socket, setSocket] = useState<Socket | null>(null);
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null);
  const [userInfo, setUserInfo] = useState<User | null>(null);
  const [connectionStatus, setConnectionStatus] = useState({ message: 'Connecting...', status: 'connecting' });
  const [cameraStatus, setCameraStatus] = useState({ message: 'Connecting...', status: 'connecting' });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [isElectron, setIsElectron] = useState(false);
  const [browserInfo, setBrowserInfo] = useState({ platform: '', userAgent: '' });
  const [showFaceAuth, setShowFaceAuth] = useState(false);
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Check if we're running in Electron
    const electronFlag = typeof window !== 'undefined' && (window as any).isElectron;
    setIsElectron(electronFlag);
    
    // Get browser info safely
    if (typeof window !== 'undefined') {
      setBrowserInfo({
        platform: navigator.platform || 'Unknown',
        userAgent: navigator.userAgent || 'Unknown'
      });
    }
    
    initializeSpotify();
    initializeWebSocket();
    displaySystemInfo();
    
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  const initializeWebSocket = () => {
    try {
      const newSocket = io(WEBSOCKET_URL);
      
      newSocket.on('connect', () => {
        console.log('🔌 Connected to camera control WebSocket');
        showNotification('📹 Camera Control Connected', 'Gesture detection is now active!');
        setCameraStatus({ message: 'Connected', status: 'connected' });
        newSocket.emit('request_playback_state');
      });
      
      newSocket.on('disconnect', () => {
        console.log('🔌 Disconnected from camera control WebSocket');
        showNotification('📹 Camera Control Disconnected', 'Gesture detection is offline');
        setCameraStatus({ message: 'Disconnected', status: 'error' });
      });
      
      newSocket.on('playback_state_changed', (data: any) => {
        console.log('🎵 Playback state changed via camera:', data);
        setIsPlaying(data.is_playing);
        const action = data.is_playing ? '▶️ Playing' : '⏸️ Paused';
        showNotification('📹 Camera Control', `Music ${action} via gesture detection`);
        setTimeout(updateCurrentTrack, 500);
      });
      
      newSocket.on('playback_state', (data: any) => {
        console.log('🎵 Current playback state:', data);
        if (!data.error) {
          setIsPlaying(data.is_playing);
        }
      });
      
      newSocket.on('connected', (data: any) => {
        console.log('📹 Camera control system message:', data.message);
      });
      
      setSocket(newSocket);
    } catch (error) {
      console.error('WebSocket connection error:', error);
      showNotification('❌ WebSocket Error', 'Could not connect to camera control system');
    }
  };

  const initializeSpotify = async () => {
    try {
      setConnectionStatus({ message: 'Connecting to Spotify...', status: 'connecting' });
      
      const userResponse = await fetch(`${SPOTIFY_API_BASE}/user`);
      if (userResponse.ok) {
        const userData = await userResponse.json();
        if (userData.success) {
          setUserInfo(userData.user);
          setConnectionStatus({ message: 'Connected to Spotify', status: 'connected' });
          startPlaybackTracking();
          showNotification('🎵 Connected to Spotify!', `Welcome back, ${userData.user.name}!`);
        } else {
          throw new Error(userData.error || 'Failed to get user info');
        }
      } else {
        throw new Error('Failed to connect to Spotify API');
      }
    } catch (error) {
      console.error('Spotify connection error:', error);
      setConnectionStatus({ message: 'Connection failed', status: 'error' });
      showNotification('❌ Spotify Connection Failed', 'Make sure the backend server is running and Spotify is authorized.');
    }
  };

  const startPlaybackTracking = () => {
    updateCurrentTrack();
    updateIntervalRef.current = setInterval(updateCurrentTrack, 2000);
  };

  const updateCurrentTrack = async () => {
    try {
      const response = await fetch(`${SPOTIFY_API_BASE}/current`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          if (data.track) {
            setCurrentTrack(data.track);
            setIsPlaying(data.track.is_playing);
          }
        }
      }
    } catch (error) {
      console.error('Update track error:', error);
    }
  };

  const togglePlayPause = async () => {
    try {
      const response = await fetch(`${SPOTIFY_API_BASE}/play`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setIsPlaying(data.action === 'playing');
          setTimeout(updateCurrentTrack, 500);
        }
      }
    } catch (error) {
      console.error('Play/pause error:', error);
      showNotification('❌ Playback Error', 'Failed to control playback');
    }
  };

  const previousTrack = async () => {
    try {
      const response = await fetch(`${SPOTIFY_API_BASE}/previous`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        setTimeout(updateCurrentTrack, 500);
      }
    } catch (error) {
      console.error('Previous track error:', error);
    }
  };

  const nextTrack = async () => {
    try {
      const response = await fetch(`${SPOTIFY_API_BASE}/next`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        setTimeout(updateCurrentTrack, 500);
      }
    } catch (error) {
      console.error('Next track error:', error);
    }
  };

  const performSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setSearchResults([]);
      const response = await fetch(`${SPOTIFY_API_BASE}/search?q=${encodeURIComponent(searchQuery)}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setSearchResults(data.tracks);
        }
      }
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  const playTrack = async (uri: string) => {
    try {
      const response = await fetch(`${SPOTIFY_API_BASE}/play-track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uri })
      });
      
      if (response.ok) {
        setTimeout(updateCurrentTrack, 1000);
        showNotification('🎵 Playing Track', 'Track started successfully!');
      }
    } catch (error) {
      console.error('Play track error:', error);
      showNotification('❌ Playback Error', 'Failed to play track');
    }
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const displaySystemInfo = async () => {
    if (isElectron && typeof window !== 'undefined' && (window as any).electronAPI) {
      try {
        const info = await (window as any).electronAPI.getSystemInfo();
        setSystemInfo(info);
      } catch (error) {
        console.error('Failed to get system info:', error);
      }
    }
  };

  const showNotification = (title: string, message: string) => {
    // Create and show notification
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: white;
      padding: 15px 20px;
      border-radius: 10px;
      box-shadow: 0 5px 20px rgba(0,0,0,0.2);
      z-index: 1000;
      transform: translateX(100%);
      transition: transform 0.3s ease;
      max-width: 300px;
    `;
    
    notification.innerHTML = `
      <h4 style="margin: 0 0 5px 0; color: #667eea;">${title}</h4>
      <p style="margin: 0; color: #666; font-size: 14px;">${message}</p>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.style.transform = 'translateX(0)';
    }, 100);
    
    setTimeout(() => {
      notification.style.transform = 'translateX(100%)';
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 5000);
  };

  const getProgressPercentage = () => {
    if (!currentTrack || currentTrack.duration_ms === 0) return 0;
    return (currentTrack.progress_ms / currentTrack.duration_ms) * 100;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* User Status Section */}
        <div className="mb-8">
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <span className="text-white text-xl">👤</span>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-800">
                    {status === 'loading' ? 'Loading...' : 
                     status === 'authenticated' ? `Welcome, ${session?.user?.name || 'User'}!` : 
                     'Not logged in'}
                  </h3>
                  <p className="text-gray-600">
                    {status === 'authenticated' ? 'Face authentication successful' : 
                     status === 'unauthenticated' ? 'Please log in with your face' : 
                     'Checking authentication...'}
                  </p>
                </div>
              </div>
              {status === 'authenticated' && (
                <button
                  onClick={() => signOut()}
                  className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
                >
                  Sign Out
                </button>
              )}
            </div>
          </div>
        </div>

        <header className="text-center mb-12">
          <h1 className="text-6xl font-bold text-gray-800 mb-4">🚀 Jupbox</h1>
          <p className="text-xl text-gray-600">
            {isElectron ? 'Next.js + Electron + Spotify + Face Auth' : 'Next.js + Spotify + Face Auth'}
          </p>
          {isElectron && (
            <div className="mt-2 px-4 py-2 bg-blue-100 text-blue-800 rounded-full inline-block">
              🖥️ Running in Electron
            </div>
          )}
        </header>
        
        <main className="space-y-8">
          {/* Face Authentication Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <FaceLogin />
            <FaceEnrollment />
          </div>
          
          {/* Spotify Player Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold text-gray-800">🎵 Spotify Player</h2>
              <div className="flex items-center space-x-3">
                <span className="text-gray-700">{userInfo?.name || 'Loading...'}</span>
                {userInfo?.image && (
                  <img 
                    src={userInfo.image} 
                    alt="User" 
                    className="w-10 h-10 rounded-full"
                  />
                )}
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="flex items-center space-x-6">
                <div className="w-32 h-32 bg-gray-200 rounded-xl flex items-center justify-center">
                  {currentTrack?.cover_url ? (
                    <img 
                      src={currentTrack.cover_url} 
                      alt="Album Cover" 
                      className="w-full h-full rounded-xl object-cover"
                    />
                  ) : (
                    <span className="text-4xl">🎵</span>
                  )}
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-800 mb-2">
                    {currentTrack?.name || 'No track playing'}
                  </h3>
                  <p className="text-lg text-gray-600 mb-1">
                    {currentTrack?.artist || 'Connect to Spotify'}
                  </p>
                  <p className="text-gray-500">
                    {currentTrack?.album || ''}
                  </p>
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${getProgressPercentage()}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-sm text-gray-600">
                  <span>{currentTrack ? formatTime(currentTrack.progress_ms) : '0:00'}</span>
                  <span>{currentTrack ? formatTime(currentTrack.duration_ms) : '0:00'}</span>
                </div>
              </div>
              
              <div className="flex justify-center space-x-4">
                <button 
                  onClick={previousTrack}
                  className="p-3 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
                  title="Previous"
                >
                  <span className="text-2xl">⏮</span>
                </button>
                <button 
                  onClick={togglePlayPause}
                  className="p-4 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white transition-all transform hover:scale-105"
                  title="Play/Pause"
                >
                  <span className="text-3xl">{isPlaying ? '⏸' : '▶'}</span>
                </button>
                <button 
                  onClick={nextTrack}
                  className="p-3 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
                  title="Next"
                >
                  <span className="text-2xl">⏭</span>
                </button>
              </div>
              
              <div className="flex items-center justify-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  connectionStatus.status === 'connected' ? 'bg-green-500' : 
                  connectionStatus.status === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span className="text-gray-600">{connectionStatus.message}</span>
              </div>
            </div>
          </div>
          
          {/* Camera Control Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">📹 Camera Control</h2>
            <div className="space-y-4">
              <p className="text-gray-700">
                <strong>Gesture Detection:</strong> 
                <span className={`ml-2 px-3 py-1 rounded-full text-sm ${
                  cameraStatus.status === 'connected' ? 'bg-green-100 text-green-800' :
                  cameraStatus.status === 'connecting' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {cameraStatus.message}
                </span>
              </p>
              <div>
                <p className="text-gray-700 mb-2"><strong>Instructions:</strong></p>
                <ul className="list-disc list-inside space-y-1 text-gray-600">
                  <li>Show <strong>fist</strong> to <strong>PAUSE</strong> music</li>
                  <li>Show <strong>palm</strong> to <strong>PLAY</strong> music</li>
                </ul>
              </div>
            </div>
          </div>
          
          {/* Emotion Detection Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">🎭 Emotion Detection</h2>
            <EmotionDetection />
          </div>
          
          {/* Search Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">🔍 Search Tracks</h2>
            <div className="space-y-4">
              <div className="flex space-x-3">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for songs..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  onKeyPress={(e) => e.key === 'Enter' && performSearch()}
                />
                <button 
                  onClick={performSearch}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105"
                >
                  Search
                </button>
              </div>
              
              {searchResults.length > 0 && (
                <div className="space-y-3">
                  {searchResults.map((track, index) => (
                    <div key={index} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                      <img 
                        src={track.cover_url || ''} 
                        alt="Cover" 
                        className="w-12 h-12 rounded object-cover"
                        onError={(e) => (e.target as HTMLImageElement).style.display = 'none'}
                      />
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-800">{track.name}</h4>
                        <p className="text-gray-600">{track.artist} • {track.album}</p>
                      </div>
                      <button 
                        onClick={() => playTrack(track.uri)}
                        className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                      >
                        Play
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          {/* System Info Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">System Info</h2>
            <div className="space-y-3">
              <p className="text-gray-700">
                <strong>Platform:</strong> <span>{systemInfo?.platform || browserInfo.platform}</span>
              </p>
              {isElectron && systemInfo && (
                <>
                  <p className="text-gray-700">
                    <strong>Node Version:</strong> <span>{systemInfo.nodeVersion}</span>
                  </p>
                  <p className="text-gray-700">
                    <strong>Electron Version:</strong> <span>{systemInfo.electronVersion}</span>
                  </p>
                  <p className="text-gray-700">
                    <strong>App Version:</strong> <span>{systemInfo.appVersion}</span>
                  </p>
                </>
              )}
              <p className="text-gray-700">
                <strong>User Agent:</strong> <span className="text-sm">{browserInfo.userAgent}</span>
              </p>
            </div>
          </div>
        </main>
        

        
        <footer className="text-center mt-16 text-gray-600">
          <p>Built with ❤️ using {isElectron ? 'Next.js + Electron' : 'Next.js'} + Face Authentication</p>
        </footer>
      </div>
    </div>
  );
}
