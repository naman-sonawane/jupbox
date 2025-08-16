'use client';

import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import Link from 'next/link';

// API endpoints
const SPOTIFY_API_BASE = 'http://localhost:5000/api/spotify';
const WEBSOCKET_URL = 'http://localhost:5000';

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

export default function SpotifyPage() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null);
  const [userInfo, setUserInfo] = useState<User | null>(null);
  const [connectionStatus, setConnectionStatus] = useState({ message: 'Connecting...', status: 'connecting' });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gestureStatus, setGestureStatus] = useState<'idle' | 'starting' | 'running' | 'stopping' | 'error'>('idle');
  const [lastGesture, setLastGesture] = useState<string>('');
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    initializeSpotify();
    initializeWebSocket();
    
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
        console.log('🔌 Connected to Jupbox WebSocket');
        setConnectionStatus({ message: 'Connected', status: 'connected' });
        newSocket.emit('request_playback_state');
      });
      
      newSocket.on('disconnect', () => {
        console.log('🔌 Disconnected from Jupbox WebSocket');
        setConnectionStatus({ message: 'Disconnected', status: 'error' });
      });
      
      newSocket.on('playback_state_changed', (data: any) => {
        console.log('🎵 Playback state changed via gesture:', data);
        setIsPlaying(data.is_playing);
        setTimeout(updateCurrentTrack, 500);
      });
      
      newSocket.on('gesture_detected', (data: any) => {
        console.log('🎮 Gesture detected:', data);
        // Show gesture notification
        if (data.action === 'pause') {
          setIsPlaying(false);
          setLastGesture('👊 Fist detected - Music paused!');
        } else if (data.action === 'play') {
          setIsPlaying(true);
          setLastGesture('✋ Palm detected - Music playing!');
        }
        
        // Clear gesture notification after 3 seconds
        setTimeout(() => setLastGesture(''), 3000);
      });
      
      newSocket.on('playback_state', (data: any) => {
        console.log('🎵 Current playback state:', data);
        if (!data.error) {
          setIsPlaying(data.is_playing);
        }
      });
      
      newSocket.on('connected', (data: any) => {
        console.log('📹 Jupbox system message:', data.message);
      });
      
      setSocket(newSocket);
    } catch (error) {
      console.error('WebSocket connection error:', error);
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
        } else {
          throw new Error(userData.error || 'Failed to get user info');
        }
      } else {
        throw new Error('Failed to connect to Spotify API');
      }
    } catch (error) {
      console.error('Spotify connection error:', error);
      setConnectionStatus({ message: 'Connection failed', status: 'error' });
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
      }
    } catch (error) {
      console.error('Play track error:', error);
    }
  };

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getProgressPercentage = () => {
    if (!currentTrack || currentTrack.duration_ms === 0) return 0;
    return (currentTrack.progress_ms / currentTrack.duration_ms) * 100;
  };

  // Gesture Control Functions
  const startGestureDetection = async () => {
    try {
      setGestureStatus('starting');
      const response = await fetch('http://localhost:5000/api/gesture/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      if (data.success) {
        setGestureStatus('running');
        console.log('✅ Gesture detection started');
      } else {
        throw new Error(data.message || 'Failed to start gesture detection');
      }
    } catch (error) {
      console.error('Error starting gesture detection:', error);
      setGestureStatus('error');
    }
  };

  const stopGestureDetection = async () => {
    try {
      setGestureStatus('stopping');
      const response = await fetch('http://localhost:5000/api/gesture/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      if (data.success) {
        setGestureStatus('idle');
        console.log('✅ Gesture detection stopped');
      } else {
        throw new Error(data.message || 'Failed to stop gesture detection');
      }
    } catch (error) {
      console.error('Error stopping gesture detection:', error);
      setGestureStatus('error');
    }
  };

  const checkGestureStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/gesture/status');
      const data = await response.json();
      if (data.success) {
        setGestureStatus(data.running ? 'running' : 'idle');
      }
    } catch (error) {
      console.error('Error checking gesture status:', error);
    }
  };

  useEffect(() => {
    checkGestureStatus();
    const interval = setInterval(checkGestureStatus, 5000); // Check status every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center space-x-3 text-gray-600 hover:text-gray-900 transition-colors">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <span className="text-lg font-semibold">Back to Home</span>
              </Link>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-2 rounded-full bg-gray-100">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus.status === 'connected' ? 'bg-green-500' : 
                  connectionStatus.status === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm text-gray-700">{connectionStatus.message}</span>
              </div>
              {userInfo?.image && (
                <img 
                  src={userInfo.image} 
                  alt="User" 
                  className="w-10 h-10 rounded-full"
                />
              )}
              <span className="text-gray-700 font-medium">{userInfo?.name || 'Loading...'}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-6xl font-bold text-gray-800 mb-4">🎵 Spotify Control</h1>
          <p className="text-xl text-gray-600">
            Control your music with beautiful UI and gesture integration
          </p>
        </header>
        
        <main className="space-y-8">
          {/* Gesture Notification */}
          {lastGesture && (
            <div className="bg-gradient-to-r from-green-500 to-blue-500 rounded-2xl p-6 text-white text-center animate-pulse">
              <div className="text-2xl mb-2">🎮</div>
              <div className="text-xl font-bold">{lastGesture}</div>
            </div>
          )}

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
            </div>
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

          {/* Gesture Control Card */}
          <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
            <h2 className="text-3xl font-bold mb-6 text-center">🎮 Gesture Control</h2>
            <p className="text-center text-purple-100 mb-6">
              Control your music with hand gestures - no touching required!
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold mb-4">Music Control</h3>
                <ul className="space-y-3">
                  <li className="flex items-center space-x-3">
                    <span className="text-3xl">👊</span>
                    <span>Show <strong>fist</strong> to <strong>PAUSE</strong> music</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <span className="text-3xl">✋</span>
                    <span>Show <strong>palm</strong> to <strong>PLAY</strong> music</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <span className="text-3xl">✌️</span>
                    <span>Show <strong>peace sign</strong> for <strong>NEXT</strong> track</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <span className="text-3xl">👍</span>
                    <span>Show <strong>thumbs up</strong> to <strong>LIKE</strong> track</span>
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold mb-4">Tips for Best Results</h3>
                <ul className="space-y-3">
                  <li className="flex items-center space-x-3">
                    <span className="text-2xl">💡</span>
                    <span>Ensure good lighting</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <span className="text-2xl">📱</span>
                    <span>Keep hand clearly visible</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <span className="text-2xl">⏱️</span>
                    <span>Hold gesture for 1-2 seconds</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <span className="text-2xl">📷</span>
                    <span>Stay within camera view</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Gesture Control Buttons */}
            <div className="mt-8 text-center">
              <div className="bg-white/20 rounded-xl p-6">
                <h3 className="text-xl font-semibold mb-4">Gesture Detection Control</h3>
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={startGestureDetection}
                    disabled={gestureStatus === 'starting' || gestureStatus === 'running'}
                    className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                  >
                    {gestureStatus === 'starting' ? 'Starting...' : '🚀 Start Gestures'}
                  </button>
                  <button
                    onClick={stopGestureDetection}
                    disabled={gestureStatus === 'stopping' || gestureStatus === 'idle'}
                    className="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                  >
                    {gestureStatus === 'stopping' ? 'Stopping...' : '⏹️ Stop Gestures'}
                  </button>
                </div>
                <div className="mt-4 flex items-center justify-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    gestureStatus === 'running' ? 'bg-green-400' : 
                    gestureStatus === 'starting' ? 'bg-yellow-400' : 
                    gestureStatus === 'stopping' ? 'bg-orange-400' : 
                    gestureStatus === 'error' ? 'bg-red-400' : 'bg-gray-400'
                  }`}></div>
                  <span className="text-sm">
                    {gestureStatus === 'running' ? 'Active' : 
                     gestureStatus === 'starting' ? 'Starting...' : 
                     gestureStatus === 'stopping' ? 'Stopping...' : 
                     gestureStatus === 'error' ? 'Error' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </main>
        
        <footer className="text-center mt-16 text-gray-600">
          <p>🎵 Spotify Control - Part of Jupbox Music System</p>
        </footer>
      </div>
    </div>
  );
}
