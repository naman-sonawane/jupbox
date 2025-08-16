'use client';

import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';

interface EmotionResult {
  face_index: number;
  emotion: string;
  confidence: number;
  face_bbox: [number, number, number, number];
  emotion_bbox: [number, number, number, number];
}

interface EmotionResponse {
  success: boolean;
  message: string;
  emotions: EmotionResult[];
  total_faces: number;
}

const EmotionDetection: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [emotionResults, setEmotionResults] = useState<EmotionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debug logging - only log when state changes
  useEffect(() => {
    console.log('🎭 EmotionDetection state changed:', { isCapturing, emotionResults: emotionResults.length });
  }, [isCapturing, emotionResults.length]);

  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        analyzeEmotions(imageSrc);
      }
    }
  }, []);

  const analyzeEmotions = async (imageSrc: string) => {
    setIsLoading(true);
    setError(null);
    setEmotionResults([]);

    try {
      const response = await fetch('/api/emotion/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageSrc }),
      });

      // Check if response is OK
      if (!response.ok) {
        if (response.status === 404) {
          setError('Backend not running. Please start the backend server.');
        } else {
          setError(`Backend error: ${response.status} ${response.statusText}`);
        }
        return;
      }

      // Try to parse JSON response
      let data: EmotionResponse;
      try {
        data = await response.json();
      } catch (parseError) {
        console.error('Failed to parse JSON response:', parseError);
        setError('Backend returned invalid response. Please check server logs.');
        return;
      }

      if (data.success) {
        setEmotionResults(data.emotions);
        console.log('🎭 Emotion detection results:', data);
      } else {
        setError(data.message || 'No emotions detected');
      }
    } catch (err) {
      console.error('Error analyzing emotions:', err);
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Cannot connect to backend. Please start the server.');
      } else {
        setError('Failed to analyze emotions. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const startCapture = () => {
    setIsCapturing(true);
    setEmotionResults([]);
    setError(null);
    
    // Test backend connection
    testBackendConnection();
  };
  
  const testBackendConnection = async () => {
    try {
      const response = await fetch('/api/emotion/test');
      if (response.ok) {
        console.log('✅ Backend emotion detection is ready');
      } else {
        console.log('⚠️ Backend emotion detection test failed:', response.status);
      }
    } catch (err) {
      console.log('❌ Cannot reach backend emotion test endpoint');
    }
  };

  const stopCapture = () => {
    setIsCapturing(false);
  };

  const getEmotionEmoji = (emotion: string) => {
    const emojiMap: { [key: string]: string } = {
      'Happy': '😊',
      'Sad': '😢',
      'Angry': '😠',
      'Surprised': '😲',
      'Disgust': '🤢',
      'Natural': '😐'
    };
    return emojiMap[emotion] || '❓';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="w-full">
      <div className="text-center mb-6">
        <p className="text-gray-600">
          Analyze emotions from multiple faces in real-time using Roboflow AI
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera Section */}
        <div className="space-y-4">
          <div className="bg-gray-100 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-gray-700">
              📸 Live Camera Feed
            </h3>
            
            {!isCapturing ? (
              <div className="text-center">
                <button
                  onClick={startCapture}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
                >
                  🚀 Start Camera
                </button>
                                 <p className="text-sm text-gray-500 mt-2">
                   Click to start the camera for emotion detection
                 </p>
                 <button
                   onClick={testBackendConnection}
                   className="mt-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors text-sm"
                 >
                   🔍 Test Backend Connection
                 </button>
              </div>
            ) : (
              <div className="space-y-3">
                                 <div className="relative">
                   <Webcam
                     ref={webcamRef}
                     audio={false}
                     screenshotFormat="image/jpeg"
                     className="w-full rounded-lg border-2 border-gray-300"
                     onUserMedia={() => console.log('🎭 Camera access granted')}
                     onUserMediaError={(err) => {
                       console.error('🎭 Camera access error:', err);
                       setError('Camera access denied. Please allow camera permissions.');
                     }}
                   />
                  {isLoading && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                      <div className="text-white text-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-2"></div>
                        <p>Analyzing emotions...</p>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={capture}
                    disabled={isLoading}
                    className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded-lg transition-colors"
                  >
                    {isLoading ? '⏳ Processing...' : '🔍 Detect Emotions'}
                  </button>
                  
                  <button
                    onClick={stopCapture}
                    className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition-colors"
                  >
                    🛑 Stop Camera
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          <div className="bg-gray-100 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-gray-700">
              🎯 Emotion Analysis Results
            </h3>
            
            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-4">
                <p className="font-semibold">⚠️ {error}</p>
              </div>
            )}
            
            {emotionResults.length === 0 && !error && !isLoading && (
              <div className="text-center text-gray-500 py-8">
                <div className="text-4xl mb-2">😊</div>
                <p>No emotions detected yet</p>
                <p className="text-sm">Start the camera and click "Detect Emotions"</p>
              </div>
            )}
            
            {emotionResults.length > 0 && (
              <div className="space-y-3">
                <div className="text-sm text-gray-600 mb-3">
                  Found {emotionResults.length} emotion(s) across {emotionResults.length} face(s)
                </div>
                
                {emotionResults.map((result, index) => (
                  <div key={index} className="bg-white rounded-lg p-4 border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-2xl">
                          {getEmotionEmoji(result.emotion)}
                        </span>
                        <span className="font-semibold text-gray-800">
                          Face {result.face_index + 1}
                        </span>
                      </div>
                      <span className={`font-bold ${getConfidenceColor(result.confidence)}`}>
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="space-y-2 text-sm text-gray-600">
                      <div>
                        <span className="font-medium">Emotion:</span> {result.emotion}
                      </div>
                      <div>
                        <span className="font-medium">Face Position:</span> 
                        <span className="font-mono ml-1">
                          ({result.face_bbox[0]}, {result.face_bbox[1]}, {result.face_bbox[2]}x{result.face_bbox[3]})
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Emotion Region:</span>
                        <span className="font-mono ml-1">
                          ({result.emotion_bbox[0]}, {result.emotion_bbox[1]}, {result.emotion_bbox[2]}x{result.emotion_bbox[3]})
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-8 bg-blue-50 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-800 mb-2">
          💡 How to Use
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-blue-700">
          <div className="flex items-start space-x-2">
            <span className="text-blue-600 font-bold">1.</span>
            <p>Click "Start Camera" to activate your webcam</p>
          </div>
          <div className="flex items-start space-x-2">
            <span className="text-blue-600 font-bold">2.</span>
            <p>Position faces in the camera view (supports multiple people)</p>
          </div>
          <div className="flex items-start space-x-2">
            <span className="text-blue-600 font-bold">3.</span>
            <p>Click "Detect Emotions" to analyze facial expressions</p>
          </div>
        </div>
      </div>

      {/* Technical Info */}
      <div className="mt-4 text-center text-xs text-gray-500">
        <p>Powered by Roboflow emotion-detection-cwq4g/1 model • Real-time AI analysis</p>
      </div>
    </div>
  );
};

export default EmotionDetection;
