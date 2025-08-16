'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Webcam from 'react-webcam';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const webcamRef = useRef<Webcam | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [showCamera, setShowCamera] = useState(false);
  const router = useRouter();

  const handleFaceLogin = async () => {
    setLoading(true);
    setMessage("📸 Capturing your face...");
    
    try {
      const screenshot = webcamRef.current?.getScreenshot();
      if (!screenshot) {
        setMessage("❌ Could not capture image. Please try again.");
        setLoading(false);
        return;
      }

      const result = await signIn("credentials", {
        redirect: false,
        image: screenshot
      });

      if (result?.ok) {
        setMessage("✅ Login successful! Redirecting...");
        setTimeout(() => {
          router.push('/spotify');
        }, 1500);
      } else {
        setMessage("❌ Face not recognized. Please try again or sign up first.");
      }
    } catch (error) {
      console.error('Login error:', error);
      setMessage("❌ Error during login. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const startCamera = () => {
    setShowCamera(true);
    setMessage("📹 Camera activated. Position your face and click login.");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-black to-purple-900 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-block mb-4">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto">
              <span className="text-white text-2xl font-bold">J</span>
            </div>
          </Link>
          <h1 className="text-3xl font-bold text-white mb-2">Welcome Back</h1>
          <p className="text-gray-300">Sign in with your face to continue</p>
        </div>

        {/* Login Card */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20">
          {!showCamera ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-center mb-6">
                <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-3xl">👤</span>
                </div>
                <h2 className="text-xl font-semibold text-white">Face Authentication</h2>
                <p className="text-gray-300 text-sm mt-2">
                  Click below to activate your camera and sign in
                </p>
              </div>

              <button
                onClick={startCamera}
                className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105"
              >
                📹 Start Camera
              </button>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-center mb-6">
                <h2 className="text-xl font-semibold text-white mb-2">Position Your Face</h2>
                <p className="text-gray-300 text-sm">
                  Look directly at the camera and click login
                </p>
              </div>

              {/* Camera Feed */}
              <div className="relative mb-6">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  className="w-full rounded-xl border-2 border-purple-500/30"
                  style={{ height: '240px' }}
                />
                {loading && (
                  <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
                    <div className="text-white text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto mb-2"></div>
                      Processing...
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={handleFaceLogin}
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? '🔐 Signing In...' : '🔐 Sign In with Face'}
                </button>
                
                <button
                  onClick={() => setShowCamera(false)}
                  className="w-full bg-gray-600/50 text-white py-3 px-6 rounded-xl font-semibold hover:bg-gray-600/70 transition-all duration-300"
                >
                  ↩️ Back
                </button>
              </div>
            </motion.div>
          )}

          {/* Message Display */}
          {message && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-4 p-3 rounded-lg text-center ${
                message.startsWith('✅') ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 
                message.startsWith('❌') ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 
                'bg-blue-500/20 text-blue-300 border border-blue-500/30'
              }`}
            >
              {message}
            </motion.div>
          )}

          {/* Footer Links */}
          <div className="mt-6 text-center">
            <p className="text-gray-400 text-sm mb-3">
              Don't have an account?
            </p>
            <Link 
              href="/auth" 
              className="text-purple-400 hover:text-purple-300 font-medium transition-colors"
            >
              Sign up with face recognition →
            </Link>
          </div>
        </div>

        {/* Back to Home */}
        <div className="text-center mt-6">
          <Link 
            href="/" 
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            ← Back to Home
          </Link>
        </div>
      </motion.div>
    </div>
  );
}
