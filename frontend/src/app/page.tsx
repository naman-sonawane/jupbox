'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import PWAInstallPrompt from '../components/PWAInstallPrompt';

export default function HomePage() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Check if backend is reachable
    fetch('http://localhost:5000/api/health')
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-md border-b border-purple-500/20">
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
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
            Welcome to{' '}
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
              Jupbox
            </span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration.
            Experience music like never before.
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {/* Face Authentication */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <Link href="/auth" className="group block">
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300 hover:scale-105 hover:bg-white/15">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                  <span className="text-2xl">👤</span>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Face Authentication</h3>
                <p className="text-gray-300 leading-relaxed">
                  Secure access to your music system using advanced face recognition. Enroll your face and enjoy seamless, secure authentication.
                </p>
                <div className="mt-6 flex items-center text-purple-400 group-hover:text-purple-300 transition-colors">
                  <span className="font-medium">Sign Up Now</span>
                  <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>
          </motion.div>

          {/* Spotify Control with Gestures */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Link href="/spotify" className="group block">
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300 hover:scale-105 hover:bg-white/15">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                  <span className="text-2xl">🎵</span>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Spotify + Gestures</h3>
                <p className="text-gray-300 leading-relaxed">
                  Control your Spotify playback with beautiful UI and hand gestures. Play, pause, skip tracks, and search for music with simple hand movements.
                </p>
                <div className="mt-6 flex items-center text-purple-400 group-hover:text-purple-300 transition-colors">
                  <span className="font-medium">Get Started</span>
                  <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>
          </motion.div>

          {/* Emotion Detection */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <Link href="/emotions" className="group block">
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300 hover:scale-105 hover:bg-white/15">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                  <span className="text-2xl">🎭</span>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Emotion Detection</h3>
                <p className="text-gray-300 leading-relaxed">
                  Advanced AI-powered emotion recognition. Analyze facial expressions and get insights into your emotional state in real-time.
                </p>
                <div className="mt-6 flex items-center text-purple-400 group-hover:text-purple-300 transition-colors">
                  <span className="font-medium">Explore</span>
                  <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>
          </motion.div>
        </div>

        {/* Quick Stats */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-16"
        >
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20">
            <div className="text-3xl font-bold text-white mb-2">🎵</div>
            <div className="text-2xl font-bold text-white">Spotify</div>
            <div className="text-gray-300">Connected</div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20">
            <div className="text-3xl font-bold text-white mb-2">👤</div>
            <div className="text-2xl font-bold text-white">Face Auth</div>
            <div className="text-gray-300">Ready</div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20">
            <div className="text-3xl font-bold text-white mb-2">🎮</div>
            <div className="text-2xl font-bold text-white">Gestures</div>
            <div className="text-gray-300">Integrated</div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20">
            <div className="text-3xl font-bold text-white mb-2">⚡</div>
            <div className="text-2xl font-bold text-white">Real-time</div>
            <div className="text-gray-300">WebSocket</div>
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.0 }}
          className="text-center"
        >
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 max-w-2xl mx-auto">
            <h2 className="text-3xl font-bold text-white mb-4">Ready to Get Started?</h2>
            <p className="text-purple-100 mb-6">
              Choose your preferred way to interact with your music. Each feature is designed to work independently.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/auth" className="px-8 py-3 bg-white text-purple-600 rounded-xl font-semibold hover:bg-gray-100 transition-colors">
                👤 Sign Up with Face
              </Link>
              <Link href="/login" className="px-8 py-3 bg-black/20 text-white rounded-xl font-semibold hover:bg-black/30 transition-colors border border-white/30">
                🔐 Sign In
              </Link>
            </div>
          </div>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="bg-black/20 backdrop-blur-md border-t border-purple-500/20 mt-20">
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

      {/* PWA Install Prompt */}
      <PWAInstallPrompt />
    </div>
  );
}
