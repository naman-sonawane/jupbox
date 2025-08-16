'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function GesturesPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<'idle' | 'starting' | 'running' | 'stopping' | 'error'>('idle');
  const [lastGesture, setLastGesture] = useState<string>('');
  const [lastGestureTime, setLastGestureTime] = useState<Date | null>(null);
  const [gestureHistory, setGestureHistory] = useState<Array<{gesture: string, time: Date, confidence?: number}>>([]);

  const startGestureDetection = async () => {
    try {
      setStatus('starting');
      const response = await fetch('http://localhost:5000/api/gesture/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      if (data.success) {
        setIsRunning(true);
        setStatus('running');
        console.log('✅ Gesture detection started');
        // Clear previous gesture history when starting
        setGestureHistory([]);
        setLastGesture('');
        setLastGestureTime(null);
      } else {
        throw new Error(data.message || 'Failed to start gesture detection');
      }
    } catch (error) {
      console.error('Error starting gesture detection:', error);
      setStatus('error');
    }
  };

  const stopGestureDetection = async () => {
    try {
      setStatus('stopping');
      const response = await fetch('http://localhost:5000/api/gesture/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      if (data.success) {
        setIsRunning(false);
        setStatus('idle');
        console.log('✅ Gesture detection stopped');
      } else {
        throw new Error(data.message || 'Failed to stop gesture detection');
      }
    } catch (error) {
      console.error('Error stopping gesture detection:', error);
      setStatus('error');
    }
  };

  const checkStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/gesture/status');
      const data = await response.json();
      if (data.success) {
        setIsRunning(data.running);
        setStatus(data.running ? 'running' : 'idle');
        console.log('Gesture system status:', data);
      }
    } catch (error) {
      console.error('Error checking gesture status:', error);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 5000); // Check status every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (status) {
      case 'running':
        return 'bg-green-500';
      case 'starting':
        return 'bg-yellow-500';
      case 'stopping':
        return 'bg-orange-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-400';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'running':
        return 'Active';
      case 'starting':
        return 'Starting...';
      case 'stopping':
        return 'Stopping...';
      case 'error':
        return 'Error';
      default:
        return 'Inactive';
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diffInSeconds < 60) {
      return `${diffInSeconds}s ago`;
    } else if (diffInSeconds < 3600) {
      const minutes = Math.floor(diffInSeconds / 60);
      return `${minutes}m ago`;
    } else {
      const hours = Math.floor(diffInSeconds / 3600);
      return `${hours}h ago`;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-100">
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
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">🎮</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-800">Gesture Control</h1>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-6xl font-bold text-gray-800 mb-4">🎮 Gesture Control</h1>
          <p className="text-xl text-gray-600">
            Control your music with hand gestures - no touching required!
          </p>
        </header>
        
        <main className="max-w-6xl mx-auto space-y-8">
          {/* Status Display */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-3xl font-bold text-gray-800">🎮 Gesture Control</h2>
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
                <span className="text-gray-600">{getStatusText()}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">System Status</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Gesture Detection:</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      isRunning ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}>
                      {isRunning ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Camera:</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      isRunning ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}>
                      {isRunning ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Spotify:</span>
                    <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                      Connected
                    </span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Last Gesture</h3>
                {lastGesture ? (
                  <div className="space-y-2">
                    <div className="text-2xl font-bold text-blue-600">{lastGesture}</div>
                    {lastGestureTime && (
                      <div className="text-sm text-gray-500">
                        {formatTimeAgo(lastGestureTime)}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-gray-400">No gestures detected yet</div>
                )}
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="text-center">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Control Panel</h3>
              <div className="flex justify-center space-x-4">
                {!isRunning ? (
                  <button
                    onClick={startGestureDetection}
                    disabled={status === 'starting'}
                    className="px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold hover:from-green-600 hover:to-emerald-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                  >
                    {status === 'starting' ? 'Starting...' : '🚀 Start Gesture Detection'}
                  </button>
                ) : (
                  <button
                    onClick={stopGestureDetection}
                    disabled={status === 'stopping'}
                    className="px-8 py-4 bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl font-semibold hover:from-red-600 hover:to-pink-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                  >
                    {status === 'stopping' ? 'Stopping...' : '⏹️ Stop Gesture Detection'}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-6 text-center">🎯 How to Use</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-xl font-semibold mb-4">Music Control</h4>
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
                <h4 className="text-xl font-semibold mb-4">Tips for Best Results</h4>
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
          </div>

          {/* Technical Info */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h3 className="text-2xl font-bold text-gray-800 mb-6 text-center">🔧 Technical Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="text-center p-4 bg-gray-50 rounded-xl">
                <div className="text-3xl mb-2">🤖</div>
                <h4 className="font-semibold text-gray-800 mb-2">Model</h4>
                <p className="text-gray-600 text-sm">Roboflow Numbers Detection</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-xl">
                <div className="text-3xl mb-2">🌐</div>
                <h4 className="font-semibold text-gray-800 mb-2">API</h4>
                <p className="text-gray-600 text-sm">Real-time inference</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-xl">
                <div className="text-3xl mb-2">🎯</div>
                <h4 className="font-semibold text-gray-800 mb-2">Confidence</h4>
                <p className="text-gray-600 text-sm">40%+ threshold</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-xl">
                <div className="text-3xl mb-2">⚡</div>
                <h4 className="font-semibold text-gray-800 mb-2">Latency</h4>
                <p className="text-gray-600 text-sm">~1-2 seconds</p>
              </div>
            </div>
          </div>
        </main>
        
        <footer className="text-center mt-16 text-gray-600">
          <p>🎮 Gesture Control - Part of Jupbox Music System</p>
        </footer>
      </div>
    </div>
  );
}
