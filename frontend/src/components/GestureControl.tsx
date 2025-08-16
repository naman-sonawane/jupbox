'use client';

import { useState, useEffect } from 'react';

interface GestureControlProps {
  onPlaybackChange?: (isPlaying: boolean) => void;
}

export default function GestureControl({ onPlaybackChange }: GestureControlProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<'idle' | 'starting' | 'running' | 'stopping' | 'error'>('idle');
  const [lastGesture, setLastGesture] = useState<string>('');
  const [lastGestureTime, setLastGestureTime] = useState<Date | null>(null);

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
    <div className="bg-white rounded-2xl shadow-xl p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold text-gray-800">🎮 Gesture Control</h2>
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
          <span className="text-gray-600">{getStatusText()}</span>
        </div>
      </div>

      <div className="space-y-6">
        {/* Status Display */}
        <div className="bg-gray-50 rounded-xl p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">System Status</h3>
              <div className="space-y-2">
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
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Last Gesture</h3>
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

        {/* Instructions */}
        <div className="bg-blue-50 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-blue-800 mb-3">🎯 How to Use</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-blue-700 mb-2">Music Control</h4>
              <ul className="space-y-2 text-blue-600">
                <li className="flex items-center space-x-2">
                  <span className="text-2xl">👊</span>
                  <span>Show <strong>fist</strong> to <strong>PAUSE</strong> music</span>
                </li>
                <li className="flex items-center space-x-2">
                  <span className="text-2xl">✋</span>
                  <span>Show <strong>palm</strong> to <strong>PLAY</strong> music</span>
                </li>
                <li className="flex items-center space-x-2">
                  <span className="text-2xl">✌️</span>
                  <span>Show <strong>peace sign</strong> for <strong>NEXT</strong> track</span>
                </li>
                <li className="flex items-center space-x-2">
                  <span className="text-2xl">👍</span>
                  <span>Show <strong>thumbs up</strong> to <strong>LIKE</strong> track</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-blue-700 mb-2">Tips</h4>
              <ul className="space-y-2 text-blue-600">
                <li>• Ensure good lighting</li>
                <li>• Keep hand clearly visible</li>
                <li>• Hold gesture for 1-2 seconds</li>
                <li>• Stay within camera view</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Technical Info */}
        <div className="bg-gray-50 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">🔧 Technical Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <p><strong>Model:</strong> Roboflow Numbers Detection</p>
              <p><strong>API:</strong> Real-time inference</p>
              <p><strong>Confidence:</strong> 40%+ threshold</p>
            </div>
            <div>
              <p><strong>Latency:</strong> ~1-2 seconds</p>
              <p><strong>Cooldown:</strong> 2 seconds</p>
              <p><strong>WebSocket:</strong> Real-time updates</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
