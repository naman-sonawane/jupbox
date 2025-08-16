#!/usr/bin/env python3

import sys
import os
import subprocess
import time

def check_dependencies():
    try:
        import flask
        import flask_cors
        import flask_socketio
        import spotipy
        import cv2
        import numpy
        import requests
        from PIL import Image
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r backend/requirements.txt")
        return False

def main():
    print("🚀 Starting Jupbox Integrated System")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if not os.path.exists(backend_dir):
        print(f"❌ Backend directory not found: {backend_dir}")
        sys.exit(1)
    
    os.chdir(backend_dir)
    
    print("📁 Changed to backend directory")
    print("🔧 Starting integrated backend...")
    print("\n📹 Camera will start automatically")
    print("🌐 Flask server will run on http://localhost:5000")
    print("🔌 WebSocket will be available on ws://localhost:5000")
    print("\n🎮 Gesture Controls:")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running backend: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
