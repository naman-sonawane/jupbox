'use client';

import { useState, useRef, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Webcam from 'react-webcam';

interface User {
  id: string;
  name: string;
  email: string;
  musicPreferences: any[];
}

interface Mix {
  id: string;
  name: string;
  description: string;
  tracks: string[];
  participants: string[];
  createdAt: Date;
}

export default function MixPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const webcamRef = useRef<Webcam>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [identifiedUser, setIdentifiedUser] = useState<User | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [generatedMix, setGeneratedMix] = useState<Mix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mixName, setMixName] = useState('');
  const [isCreatingMix, setIsCreatingMix] = useState(false);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
      return;
    }
  }, [status, router]);

  const capturePhoto = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImage(imageSrc);
      setIsCapturing(false);
      identifyUser(imageSrc);
    }
  };

  const identifyUser = async (imageData: string) => {
    setIsProcessing(true);
    setError(null);
    
    try {
      console.log('Sending image data to identify user...');
      const response = await fetch('/api/mix/identify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('Response data:', data);
      
      if (response.ok) {
        setIdentifiedUser(data.user);
      } else {
        if (data.error === 'Internal server error' && data.details?.includes('fetch')) {
          setError('Backend server is not running. Please start the backend with: python start_integrated.py');
        } else {
          setError(data.error || 'Failed to identify user');
        }
      }
    } catch (err) {
      console.error('Error identifying user:', err);
      if (err.message?.includes('fetch')) {
        setError('Backend server is not running. Please start the backend with: python start_integrated.py');
      } else {
        setError('Network error occurred');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const generateMix = async () => {
    if (!identifiedUser || !mixName.trim()) return;
    
    setIsCreatingMix(true);
    setError(null);
    
    try {
      const response = await fetch('/api/mix/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          otherUserId: identifiedUser.id,
          mixName: mixName,
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setGeneratedMix(data.mix);
        // Redirect to Spotify page to play the mix
        router.push('/spotify?mix=' + data.mix.id);
      } else {
        setError(data.error || 'Failed to generate mix');
      }
    } catch (err) {
      setError('Network error occurred');
    } finally {
      setIsCreatingMix(false);
    }
  };

  const resetProcess = () => {
    setCapturedImage(null);
    setIdentifiedUser(null);
    setGeneratedMix(null);
    setError(null);
    setMixName('');
  };

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
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
                className="text-gray-300 hover:text-white transition-colors font-medium"
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
                href="/mix" 
                className="text-purple-300 hover:text-white transition-colors font-medium"
              >
                🎼 Mix
              </Link>
              <Link 
                href="/profile" 
                className="text-gray-300 hover:text-white transition-colors font-medium"
              >
                👤 Profile
              </Link>
              <button
                onClick={() => router.push('/login')}
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
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Create a{' '}
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
              Collaborative Mix
            </span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Take a photo of another user and let AI create a perfect blend of both your music tastes.
            Experience the magic of collaborative music discovery.
          </p>
        </motion.div>

        {/* Main Process */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Camera Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20"
          >
            <h2 className="text-2xl font-bold text-white mb-6">📸 Capture Photo</h2>
            
            {!capturedImage ? (
              <div className="space-y-4">
                <div className="relative">
                  <Webcam
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    className="w-full h-64 rounded-xl object-cover"
                  />
                  {isCapturing && (
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-xl">
                      <div className="text-white text-xl">Capturing...</div>
                    </div>
                  )}
                </div>
                
                <button
                  onClick={() => {
                    setIsCapturing(true);
                    // Add a small delay to show the capturing state
                    setTimeout(() => {
                      capturePhoto();
                    }, 500);
                  }}
                  className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300"
                >
                  📸 Take Photo
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <img
                  src={capturedImage}
                  alt="Captured"
                  className="w-full h-64 rounded-xl object-cover"
                />
                <div className="flex space-x-4">
                  <button
                    onClick={capturePhoto}
                    className="flex-1 px-4 py-2 bg-blue-500/20 text-blue-300 rounded-lg hover:bg-blue-500/30 transition-colors border border-blue-500/30"
                  >
                    🔄 Retake
                  </button>
                  <button
                    onClick={resetProcess}
                    className="flex-1 px-4 py-2 bg-gray-500/20 text-gray-300 rounded-lg hover:bg-gray-500/30 transition-colors border border-gray-500/30"
                  >
                    🗑️ Reset
                  </button>
                </div>
              </div>
            )}
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20"
          >
            <h2 className="text-2xl font-bold text-white mb-6">🎵 Mix Results</h2>
            
            {isProcessing && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-4"></div>
                <p className="text-gray-300">Identifying user and analyzing music preferences...</p>
              </div>
            )}

            {error && (
              <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4 mb-4">
                <p className="text-red-300">{error}</p>
                {error.includes('Backend server is not running') && (
                  <div className="mt-3 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
                    <p className="text-yellow-200 text-sm">
                      💡 <strong>Quick Fix:</strong> Open a new terminal in the project root and run:
                    </p>
                    <code className="block mt-2 p-2 bg-black/30 rounded text-green-300 text-xs">
                      python start_integrated.py
                    </code>
                  </div>
                )}
              </div>
            )}

            {identifiedUser && !generatedMix && (
              <div className="space-y-6">
                <div className="bg-purple-500/20 border border-purple-500/30 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">👤 User Identified</h3>
                  <p className="text-purple-200">{identifiedUser.name}</p>
                  <p className="text-purple-300 text-sm">{identifiedUser.email}</p>
                </div>

                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Enter mix name..."
                    value={mixName}
                    onChange={(e) => setMixName(e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-purple-500/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
                  />
                  
                  <button
                    onClick={generateMix}
                    disabled={!mixName.trim() || isCreatingMix}
                    className="w-full px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-xl font-semibold hover:from-green-600 hover:to-blue-600 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isCreatingMix ? '🎵 Creating Mix...' : '🎵 Generate Mix'}
                  </button>
                </div>
              </div>
            )}

            {generatedMix && (
              <div className="space-y-4">
                <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">🎵 Mix Created!</h3>
                  <p className="text-green-200">{generatedMix.name}</p>
                  <p className="text-green-300 text-sm">{generatedMix.description}</p>
                </div>
                
                <button
                  onClick={() => router.push('/spotify?mix=' + generatedMix.id)}
                  className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300"
                >
                  🎵 Play Mix on Spotify
                </button>
              </div>
            )}
          </motion.div>
        </div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-12 bg-white/5 backdrop-blur-md rounded-2xl p-8 border border-purple-500/10"
        >
          <h3 className="text-2xl font-bold text-white mb-4">🎯 How It Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📸</span>
              </div>
              <h4 className="text-lg font-semibold text-white mb-2">1. Take Photo</h4>
              <p className="text-gray-300">Capture a clear photo of the other user's face</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🤖</span>
              </div>
              <h4 className="text-lg font-semibold text-white mb-2">2. AI Analysis</h4>
              <p className="text-gray-300">AI identifies the user and analyzes both music preferences</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🎵</span>
              </div>
              <h4 className="text-lg font-semibold text-white mb-2">3. Perfect Mix</h4>
              <p className="text-gray-300">AI creates a collaborative playlist blending both tastes</p>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
