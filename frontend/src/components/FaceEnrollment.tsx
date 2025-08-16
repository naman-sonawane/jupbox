'use client';

import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';

export const FaceEnrollment: React.FC = () => {
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
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">User Information</h3>
      <div className="space-y-3">
        <input
          type="text"
          placeholder="Full Name"
          value={userInfo.name}
          onChange={(e) => setUserInfo({ ...userInfo, name: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <input
          type="email"
          placeholder="Email Address"
          value={userInfo.email}
          onChange={(e) => setUserInfo({ ...userInfo, email: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <button
        onClick={startEnrollment}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
      >
        Start Enrollment
      </button>
    </div>
  );

  const renderCaptureStep = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Face Capture</h3>
      <p className="text-gray-600">Position your face in the camera and click capture to take multiple photos.</p>
      
      <div className="relative">
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="w-full rounded-lg"
        />
        {loading && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
            <div className="text-white text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
              Capturing frames...
            </div>
          </div>
        )}
      </div>
      
      <button
        onClick={captureEnrollmentFrames}
        disabled={loading}
        className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
      >
        {loading ? 'Capturing...' : 'Capture Enrollment Frames'}
      </button>
      
      <button
        onClick={() => setEnrollmentStep('info')}
        className="w-full bg-gray-500 text-white py-2 px-4 rounded-lg hover:bg-gray-600 transition-colors"
      >
        Back
      </button>
    </div>
  );

  const renderCompleteStep = () => (
    <div className="space-y-4 text-center">
      <div className="text-green-600 text-6xl mb-4">✅</div>
      <h3 className="text-lg font-semibold text-gray-800">Enrollment Complete!</h3>
      <p className="text-gray-600">
        {userInfo.name} has been successfully enrolled in the face recognition system.
      </p>
      <p className="text-sm text-gray-500">
        You can now use face authentication to log in.
      </p>
      
      <button
        onClick={resetEnrollment}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
      >
        Enroll Another User
      </button>
    </div>
  );

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Face Enrollment</h2>
      
      {message && (
        <div className={`mb-4 p-3 rounded-lg ${
          message.startsWith('✅') ? 'bg-green-100 text-green-800' : 
          message.startsWith('❌') ? 'bg-red-100 text-red-800' : 
          'bg-blue-100 text-blue-800'
        }`}>
          {message}
        </div>
      )}
      
      {enrollmentStep === 'info' && renderInfoStep()}
      {enrollmentStep === 'capture' && renderCaptureStep()}
      {enrollmentStep === 'complete' && renderCompleteStep()}
      
      <div className="mt-6 text-sm text-gray-500">
        <p><strong>Enrollment Process:</strong></p>
        <ol className="list-decimal list-inside space-y-1 mt-2">
          <li>Enter your name and email</li>
          <li>Capture multiple face images</li>
          <li>System stores your face embeddings</li>
        </ol>
      </div>
    </div>
  );
};

export default FaceEnrollment; 