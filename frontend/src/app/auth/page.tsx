'use client';

import { useState } from 'react';
import Link from 'next/link';
import FaceLogin from '../../components/FaceLogin';
import FaceEnrollment from '../../components/FaceEnrollment';

export default function AuthPage() {
  const [activeTab, setActiveTab] = useState<'login' | 'enroll'>('login');

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
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">👤</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-800">Face Authentication</h1>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-6xl font-bold text-gray-800 mb-4">👤 Face Authentication</h1>
          <p className="text-xl text-gray-600">
            Secure access to your music system using advanced face recognition
          </p>
        </header>
        
        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-2xl shadow-lg p-2">
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('login')}
                className={`px-8 py-3 rounded-xl font-semibold transition-all duration-300 ${
                  activeTab === 'login'
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-600 text-white shadow-lg transform scale-105'
                    : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
                }`}
              >
                🔐 Login
              </button>
              <button
                onClick={() => setActiveTab('enroll')}
                className={`px-8 py-3 rounded-xl font-semibold transition-all duration-300 ${
                  activeTab === 'enroll'
                    ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg transform scale-105'
                    : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
                }`}
              >
                ✨ Enroll
              </button>
            </div>
          </div>
        </div>
        
        <main className="max-w-4xl mx-auto">
          {activeTab === 'login' ? (
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-4">🔐 Face Login</h2>
                <p className="text-gray-600">
                  Look into the camera to authenticate with your enrolled face
                </p>
              </div>
              <FaceLogin />
            </div>
          ) : (
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-4">✨ Face Enrollment</h2>
                <p className="text-gray-600">
                  Enroll your face to create a secure authentication profile
                </p>
              </div>
              <FaceEnrollment />
            </div>
          )}
        </main>
        
        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12 max-w-4xl mx-auto">
          <div className="bg-gradient-to-br from-blue-500 to-cyan-600 rounded-2xl p-6 text-white text-center">
            <div className="text-3xl mb-3">🔒</div>
            <h3 className="text-xl font-bold mb-2">Secure</h3>
            <p className="text-blue-100">Advanced face recognition with high accuracy</p>
          </div>
          <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl p-6 text-white text-center">
            <div className="text-3xl mb-3">⚡</div>
            <h3 className="text-xl font-bold mb-2">Fast</h3>
            <p className="text-green-100">Quick authentication in under 2 seconds</p>
          </div>
          <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl p-6 text-white text-center">
            <div className="text-3xl mb-3">🎯</div>
            <h3 className="text-xl font-bold mb-2">Accurate</h3>
            <p className="text-purple-100">Multi-feature analysis for reliable results</p>
          </div>
        </div>
        
        <footer className="text-center mt-16 text-gray-600">
          <p>👤 Face Authentication - Part of Jupbox Music System</p>
        </footer>
      </div>
    </div>
  );
}
