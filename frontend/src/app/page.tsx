'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function HomePage() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Check if backend is reachable
    fetch('http://localhost:5000/api/health')
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">J</span>
              </div>
              <h1 className="text-2xl font-bold text-white">Jupbox</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-3 py-2 rounded-full ${
                isConnected ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm font-medium">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
            Welcome to{' '}
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent">
              Jupbox
            </span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration.
            Experience music like never before.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {/* Face Authentication */}
          <Link href="/auth" className="group">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 hover:border-white/40 transition-all duration-300 hover:scale-105 hover:bg-white/15">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">👤</span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4">Face Authentication</h3>
              <p className="text-gray-300 leading-relaxed">
                Secure access to your music system using advanced face recognition. Enroll your face and enjoy seamless, secure authentication.
              </p>
              <div className="mt-6 flex items-center text-blue-400 group-hover:text-blue-300 transition-colors">
                <span className="font-medium">Authenticate</span>
                <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </div>
            </div>
          </Link>

          {/* Spotify Control with Gestures */}
          <Link href="/spotify" className="group">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20 hover:border-white/40 transition-all duration-300 hover:scale-105 hover:bg-white/15">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">🎵</span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4">Spotify + Gestures</h3>
              <p className="text-gray-300 leading-relaxed">
                Control your Spotify playback with beautiful UI and hand gestures. Play, pause, skip tracks, and search for music with simple hand movements.
              </p>
              <div className="mt-6 flex items-center text-green-400 group-hover:text-green-300 transition-colors">
                <span className="font-medium">Get Started</span>
                <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </div>
            </div>
          </Link>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-16">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="text-3xl font-bold text-white mb-2">🎵</div>
            <div className="text-2xl font-bold text-white">Spotify</div>
            <div className="text-gray-300">Connected</div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="text-3xl font-bold text-white mb-2">👤</div>
            <div className="text-2xl font-bold text-white">Face Auth</div>
            <div className="text-gray-300">Ready</div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="text-3xl font-bold text-white mb-2">🎮</div>
            <div className="text-2xl font-bold text-white">Gestures</div>
            <div className="text-gray-300">Integrated</div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="text-3xl font-bold text-white mb-2">⚡</div>
            <div className="text-2xl font-bold text-white">Real-time</div>
            <div className="text-gray-300">WebSocket</div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center">
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 max-w-2xl mx-auto">
            <h2 className="text-3xl font-bold text-white mb-4">Ready to Get Started?</h2>
            <p className="text-purple-100 mb-6">
              Choose your preferred way to interact with your music. Each feature is designed to work independently.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/spotify" className="px-8 py-3 bg-white text-purple-600 rounded-xl font-semibold hover:bg-gray-100 transition-colors">
                🎵 Start with Spotify
              </Link>
              <Link href="/auth" className="px-8 py-3 bg-black/20 text-white rounded-xl font-semibold hover:bg-black/30 transition-colors border border-white/30">
                👤 Face Authentication
              </Link>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black/20 backdrop-blur-md border-t border-white/10 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-400">
            <p className="text-sm">
              Jupbox - Next.js + Spotify + Face Auth + Gesture Control
            </p>
            <p className="text-xs mt-2">
              Built with modern web technologies for seamless music control
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
