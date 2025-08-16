'use client';

import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import { signIn } from "next-auth/react";

const videoConstraints = { facingMode: "user" as const };

export const FaceLogin: React.FC = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const captureAndLogin = async () => {
    setLoading(true);
    setMessage("");
    try {
      const screenshot = webcamRef.current?.getScreenshot();
      if (!screenshot) {
        setMessage("❌ Could not capture image. Make sure your webcam is allowed.");
        setLoading(false);
        return;
      }

      // signIn returns a Promise resolving to a response when redirect: false
      const res = await signIn("credentials", {
        redirect: false,
        image: screenshot
      });

      setLoading(false);
      // @ts-ignore - signIn returns different shapes; basic success check:
      if (res && (res as any).ok) {
        // Successfully signed in
        setMessage("✅ Login successful! Welcome back!");
        // Refresh the page to update the session
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      } else {
        setMessage("❌ Login failed — no face match found. Please try again or enroll first.");
      }
    } catch (err) {
      console.error(err);
      setLoading(false);
      setMessage("❌ Error during face login. Please try again.");
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">🔐 Face Authentication</h2>
      <div className="space-y-6">
        <div className="flex justify-center">
          <Webcam
            audio={false}
            ref={webcamRef}
            mirrored
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className="rounded-xl border-4 border-gray-200"
            style={{ width: 320, height: 240 }}
          />
        </div>
        <div className="text-center">
          <button 
            onClick={captureAndLogin} 
            disabled={loading}
            className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Checking..." : "Login with Face"}
          </button>
        </div>
        
        {message && (
          <div className={`text-center p-3 rounded-lg ${
            message.startsWith('✅') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {message}
          </div>
        )}
        
        <div className="text-center text-gray-600">
          <p className="text-sm">
            Position your face in the camera and click the button to authenticate
          </p>
        </div>
      </div>
    </div>
  );
};

export default FaceLogin; 