'use client';

import { useState, useRef, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Webcam from 'react-webcam';

export default function EmotionsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const webcamRef = useRef<Webcam | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [emotionResult, setEmotionResult] = useState<any>(null);
  const [showCamera, setShowCamera] = useState(false);

  useEffect(() => {
    // Check authentication
    if (status === 'unauthenticated') {
      router.push('/login');
      return;
    }
  }, [status, router]);

  const startCamera = () => {
    setShowCamera(true);
    setMessage("📹 Camera activated. Position your face and click analyze.");
    setEmotionResult(null);
  };

  const analyzeEmotion = async () => {
    if (!webcamRef.current) {
      setMessage("❌ Camera not ready. Please try again.");
      return;
    }

    setLoading(true);
    setMessage("🔍 Analyzing your emotions...");

    try {
      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) {
        setMessage("❌ Could not capture image. Please try again.");
        setLoading(false);
        return;
      }

      // Convert base64 to blob
      const base64Data = screenshot.replace(/^data:image\/jpeg;base64,/, '');
      const blob = await fetch(`data:image/jpeg;base64,${base64Data}`).then(res => res.blob());

      // Create FormData for Roboflow
      const formData = new FormData();
      formData.append('file', blob, 'emotion.jpg');

      // Call backend API for emotion detection
      const response = await fetch('http://localhost:5000/api/emotions', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setEmotionResult(data);
        setMessage("✅ Emotion analysis complete!");
      } else {
        throw new Error('Failed to analyze emotion');
      }
    } catch (error) {
      console.error('Emotion analysis error:', error);
      setMessage("❌ Error during emotion analysis. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const stopCamera = () => {
    setShowCamera(false);
    setMessage("");
    setEmotionResult(null);
  };

  const playEmotionMusic = async (emotion: string, songRecommendation: string) => {
    try {
      setMessage("🎵 Playing music for your mood...");
      
      const response = await fetch('http://localhost:5000/api/emotions/music', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ emotion, song_recommendation: songRecommendation }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessage(`✅ ${data.message}`);
        
        // Redirect to Spotify dashboard with track info
        setTimeout(() => {
          const trackInfo = encodeURIComponent(JSON.stringify({
            name: data.track.name,
            artist: data.track.artist,
            album: data.track.album,
            cover_url: data.track.cover_url
          }));
          window.location.href = `/spotify?emotion_music=${emotion.toLowerCase()}&track_info=${trackInfo}`;
        }, 2000);
      } else {
        throw new Error('Failed to play music');
      }
    } catch (error) {
      console.error('Music playback error:', error);
      setMessage("❌ Error playing music. Please try again.");
    }
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
                className="text-gray-300 hover:text-white transition-colors font-medium"
              >
                🎵 Spotify
              </Link>
              <Link 
                href="/emotions" 
                className="text-purple-300 hover:text-white transition-colors font-medium"
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
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-4">🎭 Emotion Detection</h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Advanced AI-powered emotion recognition. Analyze facial expressions and get insights into your emotional state in real-time.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Camera Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20"
          >
            <h2 className="text-2xl font-bold text-white mb-6">📹 Camera Feed</h2>
            
            {!showCamera ? (
              <div className="text-center space-y-6">
                <div className="w-32 h-32 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto">
                  <span className="text-4xl">📷</span>
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white mb-2">Ready to Analyze</h3>
                  <p className="text-gray-300 text-sm">
                    Click below to activate your camera and start emotion detection
                  </p>
                </div>
                <button
                  onClick={startCamera}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105"
                >
                  📹 Start Camera
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="relative">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    className="w-full rounded-xl border-2 border-purple-500/30"
                    style={{ height: '300px' }}
                  />
                  {loading && (
                    <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
                      <div className="text-white text-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto mb-2"></div>
                        Analyzing emotions...
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={analyzeEmotion}
                    disabled={loading}
                    className="flex-1 bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? '🔍 Analyzing...' : '🔍 Analyze Emotions'}
                  </button>
                  
                  <button
                    onClick={stopCamera}
                    className="px-6 py-3 bg-gray-600/50 text-white rounded-xl font-semibold hover:bg-gray-600/70 transition-all duration-300"
                  >
                    ⏹️ Stop
                  </button>
                </div>
              </div>
            )}
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20"
          >
            <h2 className="text-2xl font-bold text-white mb-6">🎯 Analysis Results</h2>
            
            {!emotionResult ? (
              <div className="text-center space-y-4">
                <div className="w-24 h-24 bg-gray-800 rounded-full flex items-center justify-center mx-auto">
                  <span className="text-3xl">🎭</span>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">No Analysis Yet</h3>
                  <p className="text-gray-300 text-sm">
                    Start the camera and analyze your emotions to see results here
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Primary Emotion */}
                <div className="text-center p-6 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-500/30">
                  <h3 className="text-lg font-semibold text-white mb-2">Primary Emotion</h3>
                  <div className="text-4xl mb-2">
                    {emotionResult.emoji || '😐'}
                  </div>
                  <p className="text-2xl font-bold text-white capitalize">
                    {emotionResult.primary_emotion}
                  </p>
                  <p className="text-purple-300 text-sm">
                    Confidence: {Math.round(emotionResult.confidence * 100)}%
                  </p>
                </div>

                {/* Emotion Breakdown */}
                {emotionResult.emotions && (
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Emotion Breakdown</h4>
                    <div className="space-y-3">
                      {Object.entries(emotionResult.emotions).map(([emotion, confidence]: [string, any]) => (
                        <div key={emotion} className="flex items-center justify-between">
                          <span className="text-gray-300 capitalize">{emotion}</span>
                          <div className="flex items-center space-x-3">
                            <div className="w-24 bg-gray-800 rounded-full h-2">
                              <div 
                                className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${confidence * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-purple-300 text-sm w-12 text-right">
                              {Math.round(confidence * 100)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Additional Insights */}
                {emotionResult.insights && (
                  <div className="p-4 bg-gray-800/50 rounded-xl">
                    <h4 className="text-lg font-semibold text-white mb-3">💡 Insights</h4>
                    <p className="text-gray-300 text-sm">
                      {emotionResult.insights}
                    </p>
                  </div>
                )}

                {/* AI Song Recommendation */}
                {emotionResult.ai_song_recommendation && (
                  <div className="p-4 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-500/30">
                    <h4 className="text-lg font-semibold text-white mb-3">🎵 AI Song Recommendation</h4>
                    <p className="text-purple-300 text-lg font-medium mb-3">
                      "{emotionResult.ai_song_recommendation}"
                    </p>
                    <button
                      onClick={() => playEmotionMusic(emotionResult.primary_emotion, emotionResult.ai_song_recommendation)}
                      className="w-full bg-gradient-to-r from-green-500 to-emerald-500 text-white py-2 px-4 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-600 transition-all duration-300 transform hover:scale-105"
                    >
                      🎵 Play This Song on Spotify
                    </button>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </div>

        {/* Message Display */}
        {message && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 max-w-2xl mx-auto"
          >
            <div className={`p-4 rounded-xl text-center ${
              message.startsWith('✅') ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 
              message.startsWith('❌') ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 
              'bg-blue-500/20 text-blue-300 border border-blue-500/30'
            }`}>
              {message}
            </div>
          </motion.div>
        )}

        {/* Info Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12"
        >
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20 text-center">
            <div className="text-3xl mb-3">🔬</div>
            <h3 className="text-xl font-bold text-white mb-2">AI-Powered</h3>
            <p className="text-gray-300 text-sm">Advanced machine learning algorithms for accurate emotion recognition</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20 text-center">
            <div className="text-3xl mb-3">⚡</div>
            <h3 className="text-xl font-bold text-white mb-2">Real-time</h3>
            <p className="text-gray-300 text-sm">Instant analysis and results in under 2 seconds</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/20 text-center">
            <div className="text-3xl mb-3">🔒</div>
            <h3 className="text-xl font-bold text-white mb-2">Private</h3>
            <p className="text-gray-300 text-sm">Your images are processed locally and never stored</p>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
