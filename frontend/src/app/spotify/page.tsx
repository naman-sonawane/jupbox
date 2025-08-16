'use client';

import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useSession, signOut } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import Link from 'next/link';
import GestureControl from '@/components/GestureControl';

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

export default function SpotifyDashboard() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [socket, setSocket] = useState<Socket | null>(null);
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null);
  const [userInfo, setUserInfo] = useState<User | null>(null);
  const [connectionStatus, setConnectionStatus] = useState({ message: 'Connecting...', status: 'connecting' });
  const [cameraStatus, setCameraStatus] = useState({ message: 'Connecting...', status: 'connecting' });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isElectron, setIsElectron] = useState(false);
  const [emotionMusicNotification, setEmotionMusicNotification] = useState<{show: boolean, message: string, track?: any, isPlaylist?: boolean}>({show: false, message: ''});
  const updateIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Check authentication
    if (status === 'unauthenticated') {
      router.push('/login');
      return;
    }

    // Check if we're running in Electron
    const electronFlag = typeof window !== 'undefined' && (window as any).isElectron;
    setIsElectron(electronFlag);
    
    initializeSpotify();
    initializeWebSocket();
    startGestureDetection();
    checkEmotionMusicStatus();
    
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      if (socket) {
        socket.disconnect();
      }
    };
  }, [status, router]);

  const initializeWebSocket = () => {
    try {
      const newSocket = io(WEBSOCKET_URL);
      
      newSocket.on('connect', () => {
        console.log('🔌 Connected to camera control WebSocket');
        setCameraStatus({ message: 'Connected', status: 'connected' });
        newSocket.emit('request_playback_state');
      });
      
      newSocket.on('disconnect', () => {
        console.log('🔌 Disconnected from camera control WebSocket');
        setCameraStatus({ message: 'Disconnected', status: 'error' });
      });
      
      newSocket.on('playback_state_changed', (data: any) => {
        console.log('🎵 Playback state changed via camera:', data);
        setIsPlaying(data.is_playing);
        setTimeout(updateCurrentTrack, 500);
      });
      
      newSocket.on('playback_state', (data: any) => {
        console.log('🎵 Current playback state:', data);
        if (!data.error) {
          setIsPlaying(data.is_playing);
        }
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

  const startGestureDetection = async () => {
    try {
      console.log('🚀 Starting gesture detection automatically...');
      const response = await fetch('http://localhost:5000/api/gesture/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      if (data.success) {
        console.log('✅ Gesture detection started automatically');
      } else {
        console.warn('⚠️ Could not start gesture detection automatically:', data.message);
      }
    } catch (error) {
      console.warn('⚠️ Could not start gesture detection automatically:', error);
    }
  };

  const checkEmotionMusicStatus = () => {
    // Check if we came from emotion detection
    const urlParams = new URLSearchParams(window.location.search);
    const emotionMusic = urlParams.get('emotion_music');
    const trackInfo = urlParams.get('track_info');
    const emotionPlaylist = urlParams.get('emotion_playlist');
    const playlistInfo = urlParams.get('playlist_info');
    
    if (emotionMusic && trackInfo) {
      try {
        const track = JSON.parse(decodeURIComponent(trackInfo));
        setEmotionMusicNotification({
          show: true,
          message: `🎵 Playing music for your ${emotionMusic} mood!`,
          track: track
        });
        
        // Clear the URL parameters
        window.history.replaceState({}, document.title, window.location.pathname);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
          setEmotionMusicNotification({show: false, message: ''});
        }, 5000);
      } catch (error) {
        console.error('Error parsing track info:', error);
      }
    }
    
    if (emotionPlaylist && playlistInfo) {
      try {
        const playlist = JSON.parse(decodeURIComponent(playlistInfo));
        setEmotionMusicNotification({
          show: true,
          message: `🎵 Playing ${playlist.total_tracks} songs for your ${emotionPlaylist} mood!`,
          track: playlist.tracks[0], // Show first track info
          isPlaylist: true
        });
        
        // Clear the URL parameters
        window.history.replaceState({}, document.title, window.location.pathname);
        
        // Auto-hide after 8 seconds (longer for playlists)
        setTimeout(() => {
          setEmotionMusicNotification({show: false, message: ''});
        }, 8000);
      } catch (error) {
        console.error('Error parsing playlist info:', error);
      }
    }
  };

  const getProgressPercentage = () => {
    if (!currentTrack || currentTrack.duration_ms === 0) return 0;
    return (currentTrack.progress_ms / currentTrack.duration_ms) * 100;
  };

  const handleLogout = async () => {
    await signOut({ redirect: false });
    router.push('/login');
  };

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-400 mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900">
      {/* Navigation Bar */}
      <nav className="bg-black/20 backdrop-blur-md border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                  <span className="text-white text-xl font-bold">J</span>
                </div>
                <span className="text-xl font-bold text-white">Jupbox</span>
              </Link>
            </div>
            
            <div className="flex items-center space-x-6">
              <Link 
                href="/spotify" 
                className="text-purple-300 hover:text-white transition-colors font-medium"
              >
                🎵 Spotify
              </Link>
              <Link 
                href="/emotions" 
                className="text-gray-300 hover:text-white transition-colors font-medium"
              >
                🎭 Emotions
              </Link>
              <Link 
                href="/profile" 
                className="text-gray-300 hover:text-white transition-colors font-medium"
              >
                👤 Profile
              </Link>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30 transition-colors border border-red-500/30"
              >
                🚪 Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-xl">👤</span>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    Welcome, {session?.user?.name || 'User'}!
                  </h3>
                  <p className="text-gray-300">
                    Face authentication successful - Control your music with gestures
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  connectionStatus.status === 'connected' ? 'bg-green-400' : 
                  connectionStatus.status === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'
                }`}></div>
                <span className="text-gray-300 text-sm">{connectionStatus.message}</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Emotion Music Notification */}
        {emotionMusicNotification.show && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-6"
          >
            <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 backdrop-blur-md rounded-xl p-4 border border-green-500/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">🎵</span>
                  <div>
                    <p className="text-green-300 font-medium">{emotionMusicNotification.message}</p>
                    {emotionMusicNotification.track && (
                      <p className="text-green-200 text-sm">
                        {emotionMusicNotification.isPlaylist ? 
                          `First track: "${emotionMusicNotification.track.name}" by ${emotionMusicNotification.track.artist}` :
                          `Now playing: "${emotionMusicNotification.track.name}" by ${emotionMusicNotification.track.artist}`
                        }
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => setEmotionMusicNotification({show: false, message: ''})}
                  className="text-green-300 hover:text-green-200 transition-colors"
                >
                  ✕
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Spotify Player Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 mb-8"
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-3xl font-bold text-white">🎵 Spotify Player</h2>
            <div className="flex items-center space-x-3">
              <span className="text-gray-300">{userInfo?.name || 'Loading...'}</span>
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
              <div className="w-32 h-32 bg-gray-800 rounded-xl flex items-center justify-center">
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
                <h3 className="text-2xl font-bold text-white mb-2">
                  {currentTrack?.name || 'No track playing'}
                </h3>
                <p className="text-lg text-gray-300 mb-1">
                  {currentTrack?.artist || 'Connect to Spotify'}
                </p>
                <p className="text-gray-400">
                  {currentTrack?.album || ''}
                </p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${getProgressPercentage()}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-sm text-gray-400">
                <span>{currentTrack ? formatTime(currentTrack.progress_ms) : '0:00'}</span>
                <span>{currentTrack ? formatTime(currentTrack.duration_ms) : '0:00'}</span>
              </div>
            </div>
            
            <div className="flex justify-center space-x-4">
              <button 
                onClick={previousTrack}
                className="p-3 rounded-full bg-gray-800 hover:bg-gray-700 transition-colors text-white"
                title="Previous"
              >
                <span className="text-2xl">⏮</span>
              </button>
              <button 
                onClick={togglePlayPause}
                className="p-4 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white transition-all transform hover:scale-105"
                title="Play/Pause"
              >
                <span className="text-3xl">{isPlaying ? '⏸' : '▶'}</span>
              </button>
              <button 
                onClick={nextTrack}
                className="p-3 rounded-full bg-gray-800 hover:bg-gray-700 transition-colors text-white"
                title="Next"
              >
                <span className="text-2xl">⏭</span>
              </button>
            </div>
          </div>
        </motion.div>
        
        {/* Camera Control Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 mb-8"
        >
          <h2 className="text-3xl font-bold text-white mb-6">📹 Camera Control</h2>
          <div className="space-y-6">
            <div className="space-y-4">
              <p className="text-gray-300">
                <strong>Gesture Detection:</strong> 
                <span className={`ml-2 px-3 py-1 rounded-full text-sm ${
                  cameraStatus.status === 'connected' ? 'bg-green-500/20 text-green-300 border border-green-500/30' :
                  cameraStatus.status === 'connecting' ? 'bg-yellow-500/20 text-yellow-300 border border-green-500/30' :
                  'bg-red-500/20 text-red-300 border border-red-500/30'
                }`}>
                  {cameraStatus.message}
                </span>
              </p>
              <div>
                <p className="text-gray-300 mb-2"><strong>Instructions:</strong></p>
                <ul className="list-disc list-inside space-y-1 text-gray-400">
                  <li>Show <strong>fist</strong> to <strong>PAUSE</strong> music</li>
                  <li>Show <strong>palm</strong> to <strong>PLAY</strong> music</li>
                </ul>
              </div>
            </div>
            
            {/* Automatic Gesture Control Status */}
            <div className="border-t border-purple-500/20 pt-6">
              <h3 className="text-xl font-semibold text-white mb-4">🎮 Auto Gesture Control Status</h3>
              <GestureControl />
            </div>
          </div>
        </motion.div>
        
        {/* Search Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20"
        >
          <h2 className="text-3xl font-bold text-white mb-6">🔍 Search Tracks</h2>
          <div className="space-y-4">
            <div className="flex space-x-3">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for songs..."
                className="flex-1 px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                onKeyPress={(e) => e.key === 'Enter' && performSearch()}
              />
              <button 
                onClick={performSearch}
                className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105"
              >
                Search
              </button>
            </div>
            
            {searchResults.length > 0 && (
              <div className="space-y-3">
                {searchResults.map((track, index) => (
                  <div key={index} className="flex items-center space-x-4 p-4 bg-gray-800/50 rounded-xl">
                    <img 
                      src={track.cover_url || ''} 
                      alt="Cover" 
                      className="w-12 h-12 rounded object-cover"
                      onError={(e) => (e.target as HTMLImageElement).style.display = 'none'}
                    />
                    <div className="flex-1">
                      <h4 className="font-semibold text-white">{track.name}</h4>
                      <p className="text-gray-300">{track.artist} • {track.album}</p>
                    </div>
                    <button 
                      onClick={() => playTrack(track.uri)}
                      className="px-4 py-2 bg-green-500/20 text-green-300 rounded-lg hover:bg-green-500/30 transition-colors border border-green-500/30"
                    >
                      Play
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </main>
    </div>
  );
}
