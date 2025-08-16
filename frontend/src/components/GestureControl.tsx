'use client';

import { useState, useEffect } from 'react';

export default function GestureControl() {
  const [lastGesture, setLastGesture] = useState<string>('');
  const [lastGestureTime, setLastGestureTime] = useState<Date | null>(null);

  // Gesture detection is now automatic, so we just need to display status
  useEffect(() => {
    // Set up polling to check for new gestures
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:5000/api/gesture/status');
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.last_gesture) {
            setLastGesture(data.last_gesture);
            setLastGestureTime(new Date());
          }
        }
      } catch (error) {
        console.log('Checking gesture status...');
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return `${Math.floor(diffInSeconds / 86400)}d ago`;
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold text-gray-800">🎮 Auto Gesture Control</h2>
        <div className="flex items-center space-x-3">
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span className="text-gray-600">Active</span>
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
                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                    Auto-Started
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Camera:</span>
                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                    Auto-Started
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
        <div className="text-center">
          <div className="inline-flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-800 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium">Auto-Started on Page Load</span>
          </div>
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
          </div>
        </div>
      </div>
    </div>
  );
}
