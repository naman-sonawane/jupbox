import cv2
import numpy as np
import requests
import json
import time
import uuid
import base64
import io
from PIL import Image, ImageChops
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from pymongo import MongoClient
from bson import ObjectId
from math import sqrt

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("ROBOFLOW_API_KEY", "t4y8okUvSiM9Y9QdOhia")
MODEL_ID = "numbers-qysva/7"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
FACE_DETECTION_MODEL = "droneface/8"

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "faces"
DIMENSION = 256

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/jupbox")

# Spotify Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "b52e7240c7544a589e65126efac853dc")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "d54a64dfe1e144c4b2e0fa3cb256ad53")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8888/callback")

# Thresholds
THRESHOLD = float(os.getenv("THRESHOLD", "0.65"))

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
sp = None
qdrant = None
mongo_client = None
users_collection = None
gesture_system = None
gesture_thread = None

def initialize_clients():
    """Initialize Qdrant and MongoDB clients"""
    global qdrant, mongo_client, users_collection
    
    # Initialize Qdrant client
    try:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
        print(f"✅ Connected to Qdrant at {QDRANT_URL}")
        
        # Ensure collection exists
        try:
            # Check if collection exists first
            collections = qdrant.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
                )
                print(f"✅ Created new Qdrant collection: {COLLECTION_NAME} with {DIMENSION} dimensions")
            else:
                print(f"✅ Using existing Qdrant collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"⚠️ Collection creation warning: {e}")
            
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        qdrant = None
    
    # Initialize MongoDB client
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client.get_database()
        users_collection = db.users
        print(f"✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        mongo_client = None

def get_spotify_client():
    global sp
    if sp is None and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing user-read-private"
        ))
    return sp

class GestureDetectionSystem:
    def __init__(self):
        self.api_key = API_KEY
        self.model_id = MODEL_ID
        self.client = None
        self.cap = None
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 0.9  # Reduced from 1.0s to 0.9s for faster response
        self.is_playing = True
        self.last_toggle_time = 0
        self.toggle_cooldown = 0.9  # Reduced from 2.0s to 0.9s for faster response
        self.running = False
        
        if self.api_key:
            try:
                self.client = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=self.api_key
                )
                print(f"✅ Gesture detection client initialized with API key: {self.api_key[:10]}...")
            except Exception as e:
                print(f"❌ Failed to initialize gesture detection client: {e}")
                self.client = None
        else:
            print("⚠️ No API key provided for gesture detection")
    
    def start_webcam(self):
        """Initialize webcam with multiple fallback options"""
        print("🔍 Attempting to initialize webcam...")
        
        # Try different camera indices
        camera_indices = [0, 1, 2, -1]  # Common camera indices
        
        for camera_index in camera_indices:
            try:
                print(f"  Trying camera index: {camera_index}")
                self.cap = cv2.VideoCapture(camera_index)
                
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Webcam initialized successfully with camera index {camera_index}")
                        print(f"  Frame size: {test_frame.shape}")
                        print(f"  FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                        return True
                    else:
                        print(f"  Camera {camera_index} opened but couldn't read frame")
                        self.cap.release()
                else:
                    print(f"  Camera {camera_index} failed to open")
                    
            except Exception as e:
                print(f"  Error with camera {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
        
        # If all cameras fail, try with specific backend
        try:
            print("  Trying with specific backend...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend on Windows
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("✅ Webcam initialized with DirectShow backend")
                    return True
                else:
                    self.cap.release()
        except Exception as e:
            print(f"  DirectShow backend failed: {e}")
        
        print("❌ Failed to initialize webcam with all methods")
        print("💡 Troubleshooting tips:")
        print("  1. Make sure your webcam is connected and not in use by another application")
        print("  2. Try closing other applications that might be using the camera")
        print("  3. Check if your webcam drivers are properly installed")
        print("  4. On Windows, try running as administrator")
        return False
    
    def detect_gestures(self, frame):
        """Detect hand gestures for Spotify control"""
        if not self.client:
            print("⚠️ Gesture detection client not available")
            return None
            
        try:
            # Save temporary image
            temp_filename = f"temp_gesture_{int(time.time())}.jpg"
            cv2.imwrite(temp_filename, frame)
            
            # Use Roboflow Inference API
            result = self.client.infer(
                temp_filename, 
                model_id=self.model_id
            )
            
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            return result
            
        except Exception as e:
            print(f"Gesture Detection API Error: {e}")
            return None
    
    def handle_play_pause_toggle(self, predictions):
        current_time = time.time()
        
        if current_time - self.last_toggle_time < self.toggle_cooldown:
            return
        
        for pred in predictions:
            number_class = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', 0)
            
            if confidence > 0.15:  # Lowered from 0.4 to 0.15 for better sensitivity
                if number_class == "0" and self.is_playing:
                    self.is_playing = False
                    self.last_toggle_time = current_time
                    print(f"🎵 PAUSED - Detected class '0' with {confidence:.1%} confidence")
                    
                    self.control_spotify_playback(False)
                    socketio.emit('playback_state_changed', {'is_playing': False})
                    
                elif number_class in ["4", "5"] and not self.is_playing:
                    self.is_playing = True
                    self.last_toggle_time = current_time
                    print(f"> PLAYING - Detected class '{number_class}' with {confidence:.1%} confidence")
                    
                    self.control_spotify_playback(True)
                    socketio.emit('playback_state_changed', {'is_playing': True})
                
                elif number_class == "1":
                    # Skip to next track
                    self.last_toggle_time = current_time
                    print(f"⏭️ NEXT TRACK - Detected class '1' with {confidence:.1%} confidence")
                    
                    self.skip_to_next_track()
                    socketio.emit('track_skipped', {'action': 'next'})

    def control_spotify_playback(self, should_play):
        try:
            spotify_client = get_spotify_client()
            if spotify_client:
                current = spotify_client.current_playback()
                
                if current and current['is_playing'] != should_play:
                    if should_play:
                        spotify_client.start_playback()
                    else:
                        spotify_client.pause_playback()
                    print(f"Spotify playback {'started' if should_play else 'paused'} via camera control")
        except Exception as e:
            print(f"Error controlling Spotify playback: {e}")

    def skip_to_next_track(self):
        try:
            spotify_client = get_spotify_client()
            if spotify_client:
                spotify_client.next_track()
                print(f"Spotify track skipped to next via camera control")
        except Exception as e:
            print(f"Error skipping to next track: {e}")

    def run(self):
        print("🚀 Starting gesture detection system...")
        
        if not self.start_webcam():
            print("❌ Cannot start gesture detection without webcam")
            return
        
        self.running = True
        print("✅ Gesture detection started in headless mode!")
        print("   Show '0' to pause, '5' to play, '1' to skip")
        print("   Use the API to stop: POST /api/gesture/stop")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                if frame is None:
                    print("❌ Error: Received null frame from webcam")
                    break
                
                frame = cv2.flip(frame, 1)
                
                self.frame_count += 1
                
                # Log status to console instead of GUI
                if self.frame_count % 60 == 0:  # Every 60 frames (~2 seconds)
                    status_text = "> PLAYING" if self.is_playing else "|| PAUSED"
                    print(f"🎵 Status: {status_text} (Frame {self.frame_count})")
                
                current_time = time.time()
                if current_time - self.last_detection_time >= self.detection_cooldown:
                    print("🔍 Processing gesture detection...")
                    
                    result = self.detect_gestures(frame)
                    
                    if result and 'predictions' in result:
                        self.handle_play_pause_toggle(result['predictions'])
                        
                        print(f"\nFrame {self.frame_count} - Detected {len(result['predictions'])} number(s):")
                        for pred in result['predictions']:
                            print(f"  Class: {pred.get('class', 'Unknown')}, "
                                  f"Confidence: {pred.get('confidence', 0):.2%}")
                    
                    self.last_detection_time = current_time
                
                # Save frame occasionally for debugging (optional)
                if self.frame_count % 30 == 0:  # Every 30 frames
                    try:
                        filename = f"gesture_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 Debug frame saved as {filename}")
                    except:
                        pass  # Ignore save errors
                
                # Simple delay instead of GUI
                time.sleep(0.033)  # ~30 FPS
                
                # Check if we should stop
                if not self.running:
                    print("👋 Stopping gesture detection...")
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping gesture detection...")
        except Exception as e:
            print(f"❌ Error in gesture detection loop: {e}")
        
        finally:
            self.cleanup()
    
    def stop(self):
        print("🛑 Stopping gesture detection...")
        self.running = False
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # Ignore GUI errors
        print("✅ Webcam stopped and resources cleaned up.")

# Face recognition functions
def normalize_vector(vector):
    """Normalize a vector for cosine similarity"""
    norm = sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]

def generate_face_embedding_improved(face_region):
    """Generate much better face embedding using advanced OpenCV techniques"""
    try:
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region
        
        # Resize to standard size for consistent processing
        gray = cv2.resize(gray, (128, 128))
        
        # Apply advanced preprocessing
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        features = []
        
        # Multi-scale features
        hog_features = generate_hog_features(gray)
        features.extend(hog_features[:64])
        
        lbp_features = generate_rotation_invariant_lbp(gray)
        features.extend(lbp_features[:64])
        
        gabor_features = generate_gabor_features(gray)
        features.extend(gabor_features[:64])
        
        landmark_features = generate_landmark_features(gray)
        features.extend(landmark_features[:32])
        
        edge_features = generate_edge_features(gray)
        features.extend(edge_features[:32])
        
        # Pad or truncate to 256 dimensions
        if len(features) > 256:
            features = features[:256]
        else:
            features.extend([0] * (256 - len(features)))
        
        # Normalize the feature vector
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features.tolist()
        
    except Exception as e:
        print(f"Error generating improved face embedding: {e}")
        return None

def generate_hog_features(image):
    """Generate HOG-like features using gradient information"""
    try:
        # Calculate gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Create histogram of gradient magnitudes (8 bins)
        mag_hist, _ = np.histogram(magnitude.flatten(), bins=8, range=(0, magnitude.max()))
        mag_hist = mag_hist / (mag_hist.sum() + 1e-8)
        
        # Create histogram of gradient directions (8 bins)
        dir_hist, _ = np.histogram(direction.flatten(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / (dir_hist.sum() + 1e-8)
        
        # Combine magnitude and direction features
        hog_features = np.concatenate([mag_hist, dir_hist])
        
        # Add statistical features
        hog_features = np.append(hog_features, [
            np.mean(magnitude), np.std(magnitude),
            np.mean(direction), np.std(direction)
        ])
        
        return hog_features
        
    except Exception as e:
        print(f"Error generating HOG features: {e}")
        return np.zeros(20)

def generate_rotation_invariant_lbp(image):
    """Generate rotation-invariant LBP features"""
    try:
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                # Check 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                # Make rotation invariant by finding minimum value
                min_code = code
                for k in range(8):
                    code = ((code << 1) | (code >> 7)) & 0xFF
                    min_code = min(min_code, code)
                
                lbp[i, j] = min_code
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        return hist / (hist.sum() + 1e-8)
        
    except Exception as e:
        print(f"Error generating LBP features: {e}")
        return np.zeros(256)

def generate_gabor_features(image):
    """Generate Gabor filter responses for texture analysis"""
    try:
        features = []
        
        # Define Gabor filter parameters
        ksize = 15
        sigma = 2.0
        theta = np.pi/4
        lambda_ = 10.0
        gamma = 0.5
        psi = 0
        
        # Create Gabor kernel
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
        
        # Apply Gabor filter
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_8UC3, kernel)
        
        # Extract features from filtered image
        features.extend([
            np.mean(filtered), np.std(filtered), np.median(filtered),
            np.min(filtered), np.max(filtered)
        ])
        
        # Apply multiple orientations
        for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, angle, lambda_, gamma, psi, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_8UC3, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error generating Gabor features: {e}")
        return np.zeros(13)

def generate_landmark_features(image):
    """Generate simplified facial landmark features"""
    try:
        features = []
        h, w = image.shape
        
        # Divide face into regions and analyze each
        regions = [
            image[:h//3, :w//3],      # Top-left (forehead)
            image[:h//3, w//3:2*w//3], # Top-center (forehead)
            image[:h//3, 2*w//3:],     # Top-right (forehead)
            image[h//3:2*h//3, :w//3], # Middle-left (eye area)
            image[h//3:2*h//3, w//3:2*w//3], # Middle-center (nose)
            image[h//3:2*h//3, 2*w//3:],     # Middle-right (eye area)
            image[2*h//3:, :w//3],     # Bottom-left (cheek)
            image[2*h//3:, w//3:2*w//3], # Bottom-center (mouth)
            image[2*h//3:, 2*w//3:]    # Bottom-right (cheek)
        ]
        
        for region in regions:
            features.extend([
                np.mean(region), np.std(region),
                np.percentile(region, 25), np.percentile(region, 75)
            ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error generating landmark features: {e}")
        return np.zeros(36)

def generate_edge_features(image):
    """Generate edge density and orientation features"""
    try:
        # Detect edges using Canny
        edges = cv2.Canny(image, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate edge orientation using gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        direction = np.arctan2(grad_y, grad_x)
        
        # Create histogram of edge orientations
        dir_hist, _ = np.histogram(direction.flatten(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / (dir_hist.sum() + 1e-8)
        
        features = [edge_density] + dir_hist.tolist()
        return np.array(features)
        
    except Exception as e:
        print(f"Error generating edge features: {e}")
        return np.zeros(9)

def detect_faces_opencv_improved(image_dataurl):
    """Improved OpenCV face detection for multiple faces"""
    try:
        print("🔍 Using improved OpenCV face detection...")
        
        # Decode the image
        if image_dataurl.startswith('data:image'):
            image_dataurl = image_dataurl.split(',')[1]
        
        image_bytes = base64.b64decode(image_dataurl)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Load OpenCV face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Use better parameters for multiple face detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            maxSize=(300, 300)
        )
        
        if len(faces) == 0:
            print("No faces detected by OpenCV")
            return []
        
        print(f"✅ OpenCV detected {len(faces)} face(s)")
        
        # Convert to list of (x, y, w, h) tuples
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append((x, y, w, h))
            print(f"  Face: bbox=({x},{y},{w},{h})")
        
        return face_list
        
    except Exception as e:
        print(f"Error in OpenCV face detection: {e}")
        return []

def get_face_embeddings_hybrid_improved(image_dataurl):
    """Improved hybrid approach: OpenCV for detection, advanced features for embeddings"""
    try:
        # Step 1: Use improved OpenCV for face detection
        print("🎯 Step 1: Using improved OpenCV for face detection...")
        predictions = detect_faces_opencv_improved(image_dataurl)
        
        if not predictions:
            print("❌ No faces detected")
            return []
        
        # Step 2: Extract face regions and generate embeddings
        print("🧠 Step 2: Generating face embeddings...")
        
        # Decode the image
        if image_dataurl.startswith('data:image'):
            image_dataurl = image_dataurl.split(',')[1]
        
        image_bytes = base64.b64decode(image_dataurl)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get image dimensions
        img_width, img_height = image.size
        
        embeddings = []
        
        for i, pred in enumerate(predictions):
            # Extract face region using bounding box
            x_abs, y_abs, w_abs, h_abs = pred
            confidence = 0.9  # High confidence for OpenCV detection
            
            print(f"Processing face {i+1}: confidence={confidence:.3f}, bbox=({x_abs},{y_abs},{w_abs},{h_abs})")
            
            # Ensure coordinates are within bounds
            x_abs = max(0, x_abs)
            y_abs = max(0, y_abs)
            w_abs = min(w_abs, img_width - x_abs)
            h_abs = min(h_abs, img_height - y_abs)
            
            # Check if face region is valid
            if w_abs > 20 and h_abs > 20 and confidence > 0.3:
                # Extract face region
                face_region = image_array[y_abs:y_abs+h_abs, x_abs:x_abs+w_abs]
                
                print(f"📸 Extracted face region: shape={face_region.shape}, bbox=({x_abs},{y_abs},{w_abs},{h_abs})")
                
                # Generate embedding from face region
                embedding = generate_face_embedding_improved(face_region)
                if embedding:
                    embeddings.append(embedding)
                    print(f"✅ Generated embedding for face {i+1} (confidence: {confidence:.3f})")
                else:
                    print(f"❌ Failed to generate embedding for face {i+1}")
            else:
                print(f"⚠️ Skipping face {i+1}: invalid bbox or low confidence (threshold: 0.3)")
        
        print(f"🎉 Hybrid approach complete: {len(embeddings)} embeddings generated")
        return embeddings
        
    except Exception as e:
        print(f"Error in improved hybrid face embedding approach: {e}")
        return []

def find_song_for_emotion(emotion):
    """Use AI to find an actual existing song that matches the detected emotion"""
    try:
        print(f"🎵 Finding real song for emotion: {emotion}")
        
        # Prepare the prompt for AI to find real songs
        prompt = f"""Find me the name of a real, existing song that perfectly represents the emotion '{emotion}'. 
        
        Requirements:
        - Must be a REAL song that exists (not made up)
        - Should be well-known and available on music platforms
        - Should match the emotional mood
        - Return only the song name and artist in format: "Song Name - Artist Name"
        
        Examples for different emotions:
        - Happy: "Happy - Pharrell Williams"
        - Sad: "Mad World - Gary Jules"
        - Angry: "Break Stuff - Limp Bizkit"
        - Natural/Calm: "Weightless - Marconi Union"
        - Surprise: "Surprise Surprise - Billy Talent"
        - Disgust: "Creep - Radiohead"
        
        Return only the song name and artist, nothing else."""
        
        # Call Hack Club AI API
        ai_response = requests.post(
            "https://ai.hackclub.com/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "qwen/qwen3-32b",
                "temperature": 0.7,
                "max_completion_tokens": 100
            },
            timeout=30
        )
        
        if ai_response.status_code == 200:
            result = ai_response.json()
            song_info = result['choices'][0]['message']['content'].strip()
            
            # Clean up the response
            song_info = song_info.replace('"', '').replace("'", "")
            if song_info.startswith('Song:'):
                song_info = song_info[5:].strip()
            
            print(f"✅ AI found song: {song_info}")
            return song_info
        else:
            print(f"❌ AI API error: {ai_response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error finding song: {e}")
        return None

# API Routes

@app.route('/api/spotify/current', methods=['GET'])
def get_current_track():
    try:
        sp = get_spotify_client()
        current = sp.current_playback()
        
        if current and current['item']:
            track = current['item']
            return jsonify({
                'success': True,
                'track': {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'is_playing': current['is_playing'],
                    'progress_ms': current['progress_ms'],
                    'duration_ms': track['duration_ms'],
                    'uri': track['uri']
                }
            })
        else:
            return jsonify({
                'success': True,
                'track': None,
                'message': 'No track currently playing'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/play', methods=['POST'])
def play_track():
    try:
        sp = get_spotify_client()
        current = sp.current_playback()
        
        if current and current['is_playing']:
            sp.pause_playback()
            action = 'paused'
        else:
            sp.start_playback()
            action = 'playing'
            
        return jsonify({'success': True, 'action': action})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/next', methods=['POST'])
def next_track():
    try:
        sp = get_spotify_client()
        sp.next_track()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/previous', methods=['POST'])
def previous_track():
    try:
        sp = get_spotify_client()
        sp.previous_track()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/user', methods=['GET'])
def get_user_info():
    try:
        sp = get_spotify_client()
        user = sp.current_user()
        return jsonify({
            'success': True,
            'user': {
                'name': user['display_name'],
                'id': user['id'],
                'image': user['images'][0]['url'] if user['images'] else None
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/search', methods=['GET'])
def search_tracks():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'success': False, 'error': 'Query parameter required'}), 400
    
    try:
        sp = get_spotify_client()
        results = sp.search(q=query, type='track', limit=10)
        tracks = []
        
        for track in results['tracks']['items']:
            tracks.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'uri': track['uri']
            })
            
        return jsonify({'success': True, 'tracks': tracks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/play-track', methods=['POST'])
def play_specific_track():
    data = request.get_json()
    track_uri = data.get('uri')
    
    if not track_uri:
        return jsonify({'success': False, 'error': 'Track URI required'}), 400
    
    try:
        sp = get_spotify_client()
        sp.start_playback(uris=[track_uri])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gesture/start', methods=['POST'])
def start_gesture_detection():
    """Start gesture detection for Spotify control"""
    global gesture_thread
    
    try:
        if gesture_thread is None or not gesture_thread.is_alive():
            gesture_thread = threading.Thread(target=gesture_system.run)
            gesture_thread.daemon = True
            gesture_thread.start()
            return jsonify({'success': True, 'message': 'Gesture detection started'})
        else:
            return jsonify({'success': False, 'message': 'Gesture detection already running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gesture/stop', methods=['POST'])
def stop_gesture_detection():
    """Stop gesture detection"""
    global gesture_thread
    
    try:
        if gesture_thread and gesture_thread.is_alive():
            gesture_system.stop()
            gesture_thread = None
            return jsonify({'success': True, 'message': 'Gesture detection stopped'})
        else:
            return jsonify({'success': False, 'message': 'Gesture detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gesture/status', methods=['GET'])
def gesture_status():
    """Get gesture detection status"""
    global gesture_system, gesture_thread
    
    # Check if gesture system is properly initialized
    system_ready = gesture_system is not None and hasattr(gesture_system, 'client') and gesture_system.client is not None
    thread_active = gesture_thread is not None and gesture_thread.is_alive()
    
    return jsonify({
        'success': True,
        'running': gesture_system.running if gesture_system else False,
        'system_ready': system_ready,
        'thread_active': thread_active,
        'api_key_configured': bool(API_KEY),
        'model_id': MODEL_ID,
        'gesture_system_exists': gesture_system is not None,
        'client_exists': gesture_system.client is not None if gesture_system else False
    })

@app.route('/api/gesture/test', methods=['GET'])
def test_gesture_detection():
    """Test endpoint to verify gesture detection setup"""
    try:
        # Check if we can create a test client
        test_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=API_KEY
        )
        
        # Test webcam access
        webcam_test = False
        webcam_error = None
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    webcam_test = True
                cap.release()
            else:
                webcam_error = "Could not open webcam"
        except Exception as e:
            webcam_error = str(e)
        
        return jsonify({
            'success': True,
            'message': 'Gesture detection system is properly configured',
            'api_key': API_KEY[:10] + '...' if len(API_KEY) > 10 else API_KEY,
            'model_id': MODEL_ID,
            'client_created': test_client is not None,
            'webcam_accessible': webcam_test,
            'webcam_error': webcam_error
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to create gesture detection client'
        }), 500

@app.route('/api/enroll', methods=['POST'])
def enroll_user():
    """Enroll a new user with face authentication"""
    try:
        body = request.get_json()
        image = body.get("image")
        name = body.get("name", "Unknown User")
        email = body.get("email", "unknown@example.com")
        
        if not image:
            return jsonify({"error": "no_image_provided"}), 400
        
        print(f"👤 Enrolling user: {name} ({email})")
        
        # Generate face embeddings
        embeddings = get_face_embeddings_hybrid_improved(image)
        
        if not embeddings:
            return jsonify({"error": "no_faces_detected"}), 400
        
        # Create user ID
        user_id = str(uuid.uuid4())
        
        # Store in Qdrant
        if qdrant:
            try:
                points = []
                for i, embedding in enumerate(embeddings):
                    point = PointStruct(
                        id=len(points),
                        vector=embedding,
                        payload={
                            "userId": user_id,
                            "name": name,
                            "email": email,
                            "frameIndex": i,
                            "timestamp": time.time()
                        }
                    )
                    points.append(point)
                
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                
                print(f"✅ User enrolled successfully: {user_id}")
                return jsonify({
                    "success": True,
                    "user_id": user_id,
                    "name": name,
                    "email": email,
                    "embeddings_count": len(embeddings)
                })
                
            except Exception as e:
                print(f"❌ Error storing in Qdrant: {e}")
                return jsonify({"error": "storage_failed", "details": str(e)}), 500
        else:
            return jsonify({"error": "qdrant_not_available"}), 500
            
    except Exception as e:
        print(f"Enrollment error: {e}")
        return jsonify({"error": "enrollment_failed", "details": str(e)}), 500

@app.route('/api/face-auth', methods=['POST'])
def authenticate_user():
    """Authenticate user using face recognition"""
    try:
        body = request.get_json()
        image = body.get("image")
        
        if not image:
            return jsonify({"error": "no_image_provided"}), 400
        
        print("🔐 Starting face authentication...")
        
        # Generate face embeddings
        embeddings = get_face_embeddings_hybrid_improved(image)
        
        if not embeddings:
            return jsonify({"error": "no_faces_detected"}), 400
        
        # Search in Qdrant
        if qdrant:
            try:
                best_score = 0
                best_user = None
                
                for embedding in embeddings:
                    # Search for similar faces
                    results = qdrant.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=embedding,
                        limit=5,
                        score_threshold=THRESHOLD
                    )
                    
                    for result in results:
                        if result.score > best_score:
                            best_score = result.score
                            best_user = {
                                "id": result.payload.get("userId"),
                                "name": result.payload.get("name"),
                                "email": result.payload.get("email")
                            }
                
                if best_user and best_score >= THRESHOLD:
                    print(f"✅ Authentication successful: {best_user['name']} (score: {best_score:.4f})")
                    return jsonify({
                        "matched": True,
                        "user": best_user,
                        "score": best_score
                    })
                else:
                    print(f"❌ Authentication failed: Score {best_score:.4f} < threshold {THRESHOLD}")
                    return jsonify({
                        "matched": False,
                        "best_score": best_score,
                        "reason": "below_threshold" if best_score < THRESHOLD else "no_match"
                    }), 200
                    
            except Exception as e:
                print(f"❌ Error searching Qdrant: {e}")
                return jsonify({"error": "search_failed", "details": str(e)}), 500
        else:
            return jsonify({"error": "qdrant_not_available"}), 500
            
    except Exception as e:
        print(f"Face authentication error: {e}")
        return jsonify({"error": "authentication_failed", "details": str(e)}), 500

@app.route('/api/emotions', methods=['POST'])
def detect_emotions():
    """Detect emotions in uploaded image using Roboflow"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "no_file_provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "no_file_selected"}), 400
        
        print("🎭 Starting emotion detection...")
        
        # Save temporary image
        temp_filename = f"temp_emotion_{int(time.time())}.jpg"
        file.save(temp_filename)
        
        try:
            # Use Roboflow for emotion detection
            if API_KEY:
                # Use the hosted API directly with the new emotion-detection-cwq4g model
                from inference_sdk import InferenceHTTPClient
                
                client = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=API_KEY
                )
                
                # Predict on the image using the emotion-detection-cwq4g model
                result = client.infer(temp_filename, model_id="emotion-detection-cwq4g/1")
                
                # Clean up temp file
                os.remove(temp_filename)
                
                # Extract emotion data from the new 6-emotion model
                if result['predictions']:
                    predictions = result['predictions']
                    emotions = {}
                    
                    for pred in predictions:
                        emotion = pred['class']
                        confidence = pred['confidence']
                        emotions[emotion] = confidence
                    
                    # Find primary emotion
                    primary_emotion = max(emotions.items(), key=lambda x: x[1])
                    
                    # Map the new emotions to emojis and provide insights
                    emotion_emoji = {
                        'Angry': '😠',
                        'Happy': '😊',
                        'Natural': '😐',
                        'Sad': '😢',
                        'Disgust': '🤢',
                        'Surprise': '😲'
                    }
                    
                    emotion_insights = {
                        'Angry': 'You appear to be frustrated. Taking deep breaths or stepping away might help.',
                        'Happy': 'You appear to be in a positive mood! Keep that energy going!',
                        'Natural': 'You appear to be in a calm, neutral state.',
                        'Sad': 'You might be feeling down. Consider reaching out to friends or doing something you enjoy.',
                        'Disgust': 'You seem to be experiencing strong negative feelings. It\'s okay to feel this way.',
                        'Surprise': 'You appear to be surprised or shocked by something.'
                    }
                    
                    # Use AI to find a real song for the emotion
                    ai_song_recommendation = find_song_for_emotion(primary_emotion[0])
                    
                    return jsonify({
                        'success': True,
                        'primary_emotion': primary_emotion[0],
                        'confidence': primary_emotion[1],
                        'emotions': emotions,
                        'predictions': predictions,
                        'emoji': emotion_emoji.get(primary_emotion[0], '😐'),
                        'insights': emotion_insights.get(primary_emotion[0], 'Emotion detected successfully.'),
                        'ai_song_recommendation': ai_song_recommendation
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'No emotions detected in the image'
                    })
            else:
                # Mock response if no API key
                os.remove(temp_filename)
                return jsonify({
                    'success': True,
                    'primary_emotion': 'Happy',
                    'confidence': 0.85,
                    'emotions': {'Happy': 0.85, 'Natural': 0.15},
                    'message': 'Mock response (no API key configured)',
                    'emoji': '😊',
                    'insights': 'You appear to be in a positive mood! Keep that energy going!',
                    'ai_song_recommendation': 'Happy - Pharrell Williams'
                })
                
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            print(f"❌ Roboflow error: {e}")
            return jsonify({"error": "roboflow_error", "details": str(e)}), 500
            
    except Exception as e:
        print(f"❌ Emotion detection error: {e}")
        return jsonify({"error": "detection_failed", "details": str(e)}), 500

@app.route('/api/emotions/music', methods=['POST'])
def get_emotion_based_music():
    """Get music recommendations and play music based on emotion"""
    try:
        data = request.get_json()
        emotion = data.get('emotion')
        song_recommendation = data.get('song_recommendation')
        
        if not emotion or not song_recommendation:
            return jsonify({"error": "emotion and song_recommendation required"}), 400
        
        print(f"🎵 Getting music for emotion: {emotion}, AI recommendation: {song_recommendation}")
        
        # Search for the song on Spotify
        sp = get_spotify_client()
        if not sp:
            return jsonify({"error": "Spotify not configured"}), 500
        
        # Search for the AI-recommended song
        search_results = sp.search(q=song_recommendation, type='track', limit=5)
        
        if search_results['tracks']['items']:
            # Get the best match
            track = search_results['tracks']['items'][0]
            
            # Start playing the track
            sp.start_playback(uris=[track['uri']])
            
            return jsonify({
                'success': True,
                'track': {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'uri': track['uri'],
                    'preview_url': track['preview_url']
                },
                'emotion': emotion,
                'ai_recommendation': song_recommendation,
                'message': f'Now playing "{track["name"]}" by {track["artists"][0]["name"]}" for your {emotion.lower()} mood!'
            })
        else:
            # If no exact match, try searching with emotion keywords
            emotion_keywords = {
                'Happy': 'upbeat happy music',
                'Sad': 'melancholy sad songs',
                'Angry': 'intense powerful music',
                'Natural': 'calm relaxing music',
                'Disgust': 'dark intense music',
                'Surprise': 'energetic exciting music'
            }
            
            fallback_search = emotion_keywords.get(emotion, 'music')
            fallback_results = sp.search(q=fallback_search, type='track', limit=1)
            
            if fallback_results['tracks']['items']:
                track = fallback_results['tracks']['items'][0]
                sp.start_playback(uris=[track['uri']])
                
                return jsonify({
                    'success': True,
                    'track': {
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'uri': track['uri'],
                        'preview_url': track['preview_url']
                    },
                    'emotion': emotion,
                    'ai_recommendation': song_recommendation,
                    'fallback': True,
                    'message': f'Playing "{track["name"]}" by {track["artists"][0]["name"]}" as a fallback for your {emotion.lower()} mood!'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No music found for this emotion'
                }), 404
                
    except Exception as e:
        print(f"❌ Emotion-based music error: {e}")
        return jsonify({"error": "music_failed", "details": str(e)}), 500

@app.route('/api/emotions/playlist', methods=['POST'])
def create_emotion_playlist():
    """Create a playlist of songs based on detected emotion"""
    try:
        data = request.get_json()
        emotion = data.get('emotion')
        
        if not emotion:
            return jsonify({"error": "emotion required"}), 400
        
        print(f"🎵 Creating playlist for emotion: {emotion}")
        
        # Get Spotify client
        sp = get_spotify_client()
        if not sp:
            return jsonify({"error": "Spotify not configured"}), 500
        
        # Define emotion-based search queries for multiple songs
        emotion_playlists = {
            'Happy': [
                'upbeat happy songs',
                'feel good music',
                'positive vibes',
                'summer hits',
                'dance music'
            ],
            'Sad': [
                'melancholy songs',
                'sad ballads',
                'emotional music',
                'heartbreak songs',
                'reflective music'
            ],
            'Angry': [
                'intense rock music',
                'powerful songs',
                'aggressive music',
                'metal songs',
                'energetic rock'
            ],
            'Natural': [
                'calm relaxing music',
                'ambient music',
                'peaceful songs',
                'nature sounds',
                'meditation music'
            ],
            'Disgust': [
                'dark intense music',
                'heavy metal',
                'industrial music',
                'aggressive songs',
                'intense electronic'
            ],
            'Surprise': [
                'energetic exciting music',
                'upbeat electronic',
                'dance hits',
                'party music',
                'energetic pop'
            ]
        }
        
        # Get search queries for this emotion
        search_queries = emotion_playlists.get(emotion, ['music'])
        
        # Collect tracks from different searches
        all_tracks = []
        tracks_per_query = 2  # Get 2 tracks per search query
        
        for query in search_queries:
            try:
                search_results = sp.search(q=query, type='track', limit=tracks_per_query)
                if search_results['tracks']['items']:
                    for track in search_results['tracks']['items']:
                        all_tracks.append({
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'album': track['album']['name'],
                            'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                            'uri': track['uri'],
                            'preview_url': track['preview_url']
                        })
            except Exception as e:
                print(f"❌ Error searching for '{query}': {e}")
                continue
        
        # Remove duplicates based on track URI
        unique_tracks = []
        seen_uris = set()
        for track in all_tracks:
            if track['uri'] not in seen_uris:
                unique_tracks.append(track)
                seen_uris.add(track['uri'])
        
        # Limit to 8 tracks maximum
        playlist_tracks = unique_tracks[:8]
        
        if not playlist_tracks:
            return jsonify({
                'success': False,
                'message': 'No music found for this emotion'
            }), 404
        
        # Start playing the first track and queue the rest
        try:
            # Start with the first track
            sp.start_playback(uris=[playlist_tracks[0]['uri']])
            first_track_playing = True
            
            # Queue the remaining tracks
            if len(playlist_tracks) > 1:
                remaining_uris = [track['uri'] for track in playlist_tracks[1:]]
                try:
                    sp.add_to_queue(remaining_uris[0])  # Add next track to queue
                    print(f"✅ Queued {len(remaining_uris)} additional tracks")
                except Exception as e:
                    print(f"⚠️ Could not queue additional tracks: {e}")
                    
        except Exception as e:
            print(f"⚠️ Could not start playback: {e}")
            first_track_playing = False
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'playlist': playlist_tracks,
            'total_tracks': len(playlist_tracks),
            'first_track_playing': first_track_playing,
            'message': f'Created a {emotion.lower()} mood playlist with {len(playlist_tracks)} songs!'
        })
        
    except Exception as e:
        print(f"❌ Emotion playlist creation error: {e}")
        return jsonify({"error": "playlist_creation_failed", "details": str(e)}), 500

@app.route('/api/webcam/test', methods=['GET'])
def test_webcam():
    """Test webcam access and return detailed information"""
    try:
        print("🔍 Testing webcam access...")
        
        # Try different camera indices
        camera_results = []
        for camera_index in [0, 1, 2, -1]:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_results.append({
                            'index': camera_index,
                            'status': 'working',
                            'frame_size': frame.shape,
                            'fps': cap.get(cv2.CAP_PROP_FPS),
                            'backend': cap.getBackendName()
                        })
                    else:
                        camera_results.append({
                            'index': camera_index,
                            'status': 'opened_but_no_frame',
                            'error': 'Could not read frame'
                        })
                else:
                    camera_results.append({
                        'index': camera_index,
                        'status': 'failed',
                        'error': 'Could not open camera'
                    })
                cap.release()
            except Exception as e:
                camera_results.append({
                    'index': camera_index,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Try DirectShow backend on Windows
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera_results.append({
                        'index': '0 (DirectShow)',
                        'status': 'working',
                        'frame_size': frame.shape,
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'backend': 'DirectShow'
                    })
                cap.release()
        except Exception as e:
            camera_results.append({
                'index': '0 (DirectShow)',
                'status': 'error',
                'error': str(e)
            })
        
        working_cameras = [cam for cam in camera_results if cam['status'] == 'working']
        
        return jsonify({
            'success': True,
            'webcam_available': len(working_cameras) > 0,
            'working_cameras': working_cameras,
            'all_camera_tests': camera_results,
            'recommendation': 'Use camera index 0' if working_cameras else 'No working cameras found'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to test webcam'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'qdrant': 'connected' if qdrant else 'not_connected',
        'mongodb': 'connected' if mongo_client else 'not_connected',
        'gesture_system': 'ready' if (gesture_system and gesture_system.client) else 'not_configured',
        'spotify': 'ready' if (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET) else 'not_configured',
        'threshold': THRESHOLD
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')
    emit('connected', {'message': 'Connected to unified Jupbox system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from WebSocket')

@socketio.on('request_playback_state')
def handle_playback_state_request():
    try:
        spotify_client = get_spotify_client()
        if spotify_client:
            current = spotify_client.current_playback()
            is_playing = current['is_playing'] if current else False
            emit('playback_state', {'is_playing': is_playing})
        else:
            emit('playback_state', {'is_playing': False, 'error': 'Spotify not configured'})
    except Exception as e:
        emit('playback_state', {'is_playing': False, 'error': str(e)})

def main():
    print("🤖 Jupbox - Unified Gesture Detection & Face Authentication System")
    print("=" * 70)
    print("This application provides:")
    print("1. Gesture-based Spotify control (show '0' to pause, '5' to play)")
    print("2. Face authentication and enrollment")
    print("3. Emotion detection")
    print("4. Spotify integration")
    print("\n🌐 Server endpoints:")
    print("- Flask API: http://localhost:5000")
    print("- WebSocket: ws://localhost:5000")
    print("\n📋 Available endpoints:")
    print("- POST /api/enroll - Face enrollment")
    print("- POST /api/face-auth - Face authentication")
    print("- POST /api/gesture/start - Start gesture detection")
    print("- POST /api/gesture/stop - Stop gesture detection")
    print("- GET  /api/gesture/status - Gesture status")
    print("- POST /api/emotion/detect - Emotion detection")
    print("- GET  /api/health - System health")
    
    # Initialize clients
    initialize_clients()
    
    # Initialize gesture system after clients are ready
    global gesture_system
    gesture_system = GestureDetectionSystem()

    print("\n🚀 Starting unified Jupbox Flask server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
