'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Webcam from 'react-webcam';

export default function AuthPage() {
  const webcamRef = useRef<Webcam | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [enrollmentStep, setEnrollmentStep] = useState<'info' | 'capture' | 'complete'>('info');
  const [userInfo, setUserInfo] = useState({ name: '', email: '' });
  const [capturedFrames, setCapturedFrames] = useState<string[]>([]);

  const captureFrame = (): string | null => {
    const screenshot = webcamRef.current?.getScreenshot();
    return screenshot || null;
  };

  const startEnrollment = () => {
    if (!userInfo.name || !userInfo.email) {
      setMessage("❌ Please fill in both name and email");
      return;
    }
    setEnrollmentStep('capture');
    setMessage("");
  };

  const captureEnrollmentFrames = async () => {
    setLoading(true);
    setMessage("📸 Capturing enrollment frames...");
    
    const frames: string[] = [];
    const frameCount = 5; // Capture 5 frames for enrollment
    
    for (let i = 0; i < frameCount; i++) {
      const frame = captureFrame();
      if (frame) {
        frames.push(frame);
        setMessage(`📸 Captured frame ${i + 1}/${frameCount}...`);
        // Wait a bit between captures
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
    
    if (frames.length >= 3) {
      setCapturedFrames(frames);
      setMessage("✅ Enrollment frames captured! Submitting enrollment...");
      submitEnrollment(frames);
    } else {
      setMessage("❌ Failed to capture enough frames. Please try again.");
      setLoading(false);
    }
  };

  const submitEnrollment = async (frames: string[]) => {
    try {
      setLoading(true);
      setMessage("🔄 Submitting enrollment...");
      
      const response = await fetch('/api/enroll', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: userInfo.name,
          email: userInfo.email,
          frames: frames
        }),
      });

      const data = await response.json();
      
      if (response.ok && data.success) {
        setEnrollmentStep('complete');
        setMessage(`✅ Enrollment successful! User ID: ${data.user_id}`);
      } else {
        setMessage(`❌ Enrollment failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Enrollment error:', error);
      setMessage('❌ Network error during enrollment');
    } finally {
      setLoading(false);
    }
  };

  const resetEnrollment = () => {
    setEnrollmentStep('info');
    setUserInfo({ name: '', email: '' });
    setCapturedFrames([]);
    setMessage("");
  };

  const renderInfoStep = () => (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <div className="text-center">
        <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-3xl">👤</span>
        </div>
        <h3 className="text-xl font-semibold text-white mb-2">User Information</h3>
        <p className="text-gray-300 text-sm">Enter your details to get started</p>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
          <input
            type="text"
            placeholder="Enter your full name"
            value={userInfo.name}
            onChange={(e) => setUserInfo({ ...userInfo, name: e.target.value })}
            className="w-full px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Email Address</label>
          <input
            type="email"
            placeholder="Enter your email address"
            value={userInfo.email}
            onChange={(e) => setUserInfo({ ...userInfo, email: e.target.value })}
            className="w-full px-4 py-3 bg-white/10 border border-purple-500/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>
      </div>
      
      <button
        onClick={startEnrollment}
        className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105"
      >
        Continue to Face Capture
      </button>
    </motion.div>
  );

  const renderCaptureStep = () => (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Face Capture</h3>
        <p className="text-gray-300 text-sm">Position your face in the camera and capture multiple photos</p>
      </div>
      
      <div className="relative">
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="w-full rounded-xl border-2 border-purple-500/30"
          style={{ height: '240px' }}
        />
        {loading && (
          <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
            <div className="text-white text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto mb-2"></div>
              Capturing frames...
            </div>
          </div>
        )}
      </div>
      
      <div className="space-y-3">
        <button
          onClick={captureEnrollmentFrames}
          disabled={loading}
          className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? '📸 Capturing...' : '📸 Capture Enrollment Frames'}
        </button>
        
        <button
          onClick={() => setEnrollmentStep('info')}
          className="w-full bg-gray-600/50 text-white py-3 px-6 rounded-xl font-semibold hover:bg-gray-600/70 transition-all duration-300"
        >
          ↩️ Back
        </button>
      </div>
    </motion.div>
  );

  const renderCompleteStep = () => (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="text-center space-y-6"
    >
      <div className="text-green-400 text-6xl mb-4">✅</div>
      <h3 className="text-2xl font-bold text-white">Enrollment Complete!</h3>
      <p className="text-gray-300">
        {userInfo.name} has been successfully enrolled in the face recognition system.
      </p>
      <p className="text-sm text-gray-400">
        You can now use face authentication to log in.
      </p>
      
      <div className="space-y-3">
        <Link
          href="/login"
          className="block w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105"
        >
          🔐 Sign In Now
        </Link>
        
        <button
          onClick={resetEnrollment}
          className="w-full bg-gray-600/50 text-white py-3 px-6 rounded-xl font-semibold hover:bg-gray-600/70 transition-all duration-300"
        >
          👤 Enroll Another User
        </button>
      </div>
    </motion.div>
  );

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
          <h1 className="text-3xl font-bold text-white mb-2">Join Jupbox</h1>
          <p className="text-gray-300">Create your account with face recognition</p>
        </div>

        {/* Auth Card */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-purple-500/20">
          {enrollmentStep === 'info' && renderInfoStep()}
          {enrollmentStep === 'capture' && renderCaptureStep()}
          {enrollmentStep === 'complete' && renderCompleteStep()}
          
          {/* Message Display */}
          {message && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-6 p-3 rounded-lg text-center ${
                message.startsWith('✅') ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 
                message.startsWith('❌') ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 
                'bg-blue-500/20 text-blue-300 border border-blue-500/30'
              }`}
            >
              {message}
            </motion.div>
          )}

          {/* Footer Links */}
          {enrollmentStep === 'info' && (
            <div className="mt-6 text-center">
              <p className="text-gray-400 text-sm mb-3">
                Already have an account?
              </p>
              <Link 
                href="/login" 
                className="text-purple-400 hover:text-purple-300 font-medium transition-colors"
              >
                Sign in with face recognition →
              </Link>
            </div>
          )}
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
