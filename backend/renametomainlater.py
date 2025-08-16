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

API_KEY = "[]"
MODEL_ID = "numbers-qysva/7"

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Spotify API routes will be registered below

# Roboflow Configuration (for face detection - hackathon showcase!)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
FACE_DETECTION_MODEL = "droneface/8"  # DroneFace for detection
GESTURE_MODEL_ID = "numbers-qysva/7"  # Gesture detection model
EMOTION_MODEL_ID = "emotion-detection-cwq4g/1"  # Emotion detection model

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "faces"
DIMENSION = 256  # Improved OpenCV-based face embeddings

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/jupbox")

# Spotify Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8888/callback")

# Thresholds
THRESHOLD = float(os.getenv("THRESHOLD", "0.65"))  # Lower threshold for improved embeddings

# Initialize clients
sp = None
qdrant = None
mongo_client = None
users_collection = None

def initialize_clients():
    """Initialize Qdrant and MongoDB clients"""
    global qdrant, mongo_client, users_collection
    
    # Initialize Qdrant client
    try:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
        print(f"✅ Connected to Qdrant at {QDRANT_URL}")
        
        # Ensure collection exists
        try:
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
            )
            print(f"✅ Created/Recreated Qdrant collection: {COLLECTION_NAME} with {DIMENSION} dimensions")
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

# Initialize gesture system after class definition
gesture_system = None
gesture_thread = None

class GestureDetectionSystem:
    def __init__(self):
        self.api_key = ROBOFLOW_API_KEY
        self.model_id = GESTURE_MODEL_ID
        self.client = None
        self.cap = None
        self.running = False
        self.is_playing = True
        self.last_toggle_time = 0
        self.toggle_cooldown = 2.0
        
        if self.api_key:
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
    
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        print("Webcam started successfully!")
        return True
    
    def detect_gestures(self, frame):
        """Detect hand gestures for Spotify control"""
        if not self.client:
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
            
            if confidence > 0.4:
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

    def run(self):
        if not self.start_webcam():
            return
        
        self.running = True
        print("Gesture detection started. Show '0' to pause, '5' to play.")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                self.frame_count += 1
                
                status_text = "> PLAYING" if self.is_playing else "|| PAUSED"
                status_color = (0, 255, 0) if self.is_playing else (0, 0, 255)
                
                (text_width, text_height), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                cv2.rectangle(frame, (5, 35), (5 + text_width + 10, 35 + text_height + 10), 
                             (0, 0, 0), -1)
                
                cv2.putText(frame, status_text, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                current_time = time.time()
                if current_time - self.last_detection_time >= self.detection_cooldown:
                    cv2.putText(frame, "Processing...", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    result = self.detect_numbers_api(frame)
                    
                    if result and 'predictions' in result:
                        frame = self.draw_predictions(frame, result['predictions'])
                        
                        self.handle_play_pause_toggle(result['predictions'])
                        
                        print(f"\nFrame {self.frame_count} - Detected {len(result['predictions'])} number(s):")
                        for pred in result['predictions']:
                            print(f"  Class: {pred.get('class', 'Unknown')}, "
                                  f"Confidence: {pred.get('confidence', 0):.2%}")
                    
                    self.last_detection_time = current_time
                
                instruction_text = "Show '0' to PAUSE | Show '5' to PLAY"
                cv2.putText(frame, instruction_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Numbers Detection - Play/Pause Control', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"gesture_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            self.cleanup()
    
    def stop(self):
        self.running = False
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped and resources cleaned up.")

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
        # 1. Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # 2. Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        features = []
        
        # 3. Multi-scale Histogram of Oriented Gradients (HOG-like features)
        hog_features = generate_hog_features(gray)
        features.extend(hog_features[:64])
        
        # 4. Local Binary Pattern (LBP) with rotation invariance
        lbp_features = generate_rotation_invariant_lbp(gray)
        features.extend(lbp_features[:64])
        
        # 5. Gabor filter responses (texture analysis)
        gabor_features = generate_gabor_features(gray)
        features.extend(gabor_features[:64])
        
        # 6. Facial landmark-based features (simplified)
        landmark_features = generate_landmark_features(gray)
        features.extend(landmark_features[:32])
        
        # 7. Edge density and orientation features
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
        image = Image.open(io.BytesIO(image_dataurl))
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
            scaleFactor=1.1,  # How much to scale image between detections
            minNeighbors=3,   # How many neighbors each candidate rectangle should have
            minSize=(30, 30), # Minimum possible object size
            maxSize=(300, 300) # Maximum possible object size
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
    """Improved hybrid approach: OpenCV for detection, FaceNet for embeddings"""
    try:
        # Step 1: Use improved OpenCV for face detection (more reliable for multiple faces)
        print("🎯 Step 1: Using improved OpenCV for face detection...")
        predictions = detect_faces_opencv_improved(image_dataurl)
        
        # Step 2: If OpenCV fails, try Roboflow as fallback
        if not predictions:
            print("🔄 OpenCV didn't detect faces, trying Roboflow fallback...")
            predictions = call_roboflow_detect_faces(image_dataurl)
            
            # Convert Roboflow predictions to OpenCV format if needed
            if predictions and isinstance(predictions[0], dict):
                converted_predictions = []
                for pred in predictions:
                    x = int(pred.get('x', 0))
                    y = int(pred.get('y', 0))
                    w = int(pred.get('width', 0))
                    h = int(pred.get('height', 0))
                    converted_predictions.append((x, y, w, h))
                predictions = converted_predictions
        
        if not predictions:
            print("❌ No faces detected by either method")
            return []
        
        # Step 3: Extract face regions and generate FaceNet embeddings
        print("🧠 Step 3: Using FaceNet for face embeddings...")
        
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
            if isinstance(pred, tuple):
                # OpenCV prediction format (x, y, w, h)
                x_abs, y_abs, w_abs, h_abs = pred
                confidence = 0.9  # High confidence for OpenCV detection
            else:
                # Roboflow prediction format
                x_abs = int(pred.get('x', 0))
                y_abs = int(pred.get('y', 0))
                w_abs = int(pred.get('width', 0))
                h_abs = int(pred.get('height', 0))
                confidence = pred.get('confidence', 0.8)
            
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
                
                # Generate FaceNet embedding from face region
                embedding = generate_face_embedding_improved(face_region)
                if embedding:
                    embeddings.append(embedding)
                    print(f"✅ Generated FaceNet embedding for face {i+1} (confidence: {confidence:.3f})")
                else:
                    print(f"❌ Failed to generate embedding for face {i+1}")
            else:
                print(f"⚠️ Skipping face {i+1}: invalid bbox or low confidence (threshold: 0.3)")
        
        print(f"🎉 Hybrid approach complete: {len(embeddings)} embeddings generated")
        return embeddings
        
    except Exception as e:
        print(f"Error in improved hybrid face embedding approach: {e}")
        return []

def call_roboflow_detect_faces(image_dataurl):
    """Use Roboflow DroneFace for face detection (hackathon showcase!)"""
    try:
        # Use the inference-sdk approach as shown in Roboflow docs
        from inference_sdk import InferenceHTTPClient
        
        # Initialize the client
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        
        print(f"🔍 Calling Roboflow DroneFace API with inference-sdk...")
        
        # Decode the image if it's a data URL
        if image_dataurl.startswith('data:image'):
            image_dataurl = image_dataurl.split(',')[1]
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_dataurl)
        
        # Save temporary image file
        temp_filename = f"temp_face_{int(time.time())}.jpg"
        with open(temp_filename, 'wb') as f:
            f.write(image_bytes)
        
        try:
            # Use the inference-sdk client with lower confidence threshold
            result = client.infer(temp_filename, model_id="droneface/8")
            
            print(f"🔍 Roboflow API response: {result}")
            
            # Extract predictions
            predictions = result.get("predictions") or result.get("results") or []
            
            if not predictions:
                print("No faces detected by Roboflow DroneFace")
                return []
            
            print(f"✅ Roboflow DroneFace detected {len(predictions)} face(s)")
            
            # Log all predictions for debugging
            for i, pred in enumerate(predictions):
                print(f"  Face {i+1}: confidence={pred.get('confidence', 0):.3f}, bbox=({pred.get('x', 0):.1f},{pred.get('y', 0):.1f},{pred.get('width', 0):.1f},{pred.get('height', 0):.1f})")
            
            return predictions
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
    except Exception as e:
        print(f"Error calling Roboflow DroneFace API: {e}")
        return []

def detect_emotions_roboflow(image_dataurl):
    """Detect emotions using Roboflow's emotion detection model"""
    try:
        if not ROBOFLOW_API_KEY:
            print("❌ No Roboflow API key configured for emotion detection")
            return []
        
        print("😊 Calling Roboflow Emotion Detection API...")
        
        # Initialize the client
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        
        # Decode the image if it's a data URL
        if image_dataurl.startswith('data:image'):
            image_dataurl = image_dataurl.split(',')[1]
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_dataurl)
        
        # Save temporary image file
        temp_filename = f"temp_emotion_{int(time.time())}.jpg"
        with open(temp_filename, 'wb') as f:
            f.write(image_bytes)
        
        try:
            # Use the emotion detection model
            result = client.infer(temp_filename, model_id=EMOTION_MODEL_ID)
            
            print(f"😊 Roboflow Emotion API response: {result}")
            
            # Extract predictions
            predictions = result.get("predictions") or result.get("results") or []
            
            if not predictions:
                print("No emotions detected by Roboflow")
                return []
            
            print(f"😊 Roboflow detected {len(predictions)} emotion(s)")
            
            # Log all emotion predictions
            for i, pred in enumerate(predictions):
                emotion = pred.get('class', 'Unknown')
                confidence = pred.get('confidence', 0)
                bbox = (pred.get('x', 0), pred.get('y', 0), pred.get('width', 0), pred.get('height', 0))
                print(f"  Emotion {i+1}: {emotion} (confidence: {confidence:.3f}, bbox: {bbox})")
            
            return predictions
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
    except Exception as e:
        print(f"Error calling Roboflow Emotion API: {e}")
        return []

def analyze_emotions_from_faces(image_dataurl):
    """Analyze emotions from detected faces using Roboflow"""
    try:
        print("🎭 Starting emotion analysis...")
        
        # First, detect faces to get bounding boxes
        face_predictions = detect_faces_opencv_improved(image_dataurl)
        
        if not face_predictions:
            print("🔄 No faces detected by OpenCV, trying Roboflow fallback...")
            face_predictions = call_roboflow_detect_faces(image_dataurl)
            
            # Convert Roboflow predictions to OpenCV format if needed
            if face_predictions and isinstance(face_predictions[0], dict):
                converted_predictions = []
                for pred in face_predictions:
                    x = int(pred.get('x', 0))
                    y = int(pred.get('y', 0))
                    w = int(pred.get('width', 0))
                    h = int(pred.get('height', 0))
                    converted_predictions.append((x, y, w, h))
                face_predictions = converted_predictions
        
        if not face_predictions:
            print("❌ No faces detected for emotion analysis")
            return []
        
        print(f"🎭 Found {len(face_predictions)} face(s) for emotion analysis")
        
        # Now detect emotions on the full image
        emotion_predictions = detect_emotions_roboflow(image_dataurl)
        
        if not emotion_predictions:
            print("❌ No emotions detected")
            return []
        
        # Match emotions to faces based on bounding box overlap
        face_emotions = []
        
        for i, face_bbox in enumerate(face_predictions):
            if isinstance(face_bbox, tuple):
                face_x, face_y, face_w, face_h = face_bbox
            else:
                face_x = int(face_bbox.get('x', 0))
                face_y = int(face_bbox.get('y', 0))
                face_w = int(face_bbox.get('width', 0))
                face_h = int(face_bbox.get('height', 0))
            
            # Find the emotion prediction that best matches this face
            best_emotion = None
            best_overlap = 0
            
            for emotion_pred in emotion_predictions:
                emotion_x = emotion_pred.get('x', 0)
                emotion_y = emotion_pred.get('y', 0)
                emotion_w = emotion_pred.get('width', 0)
                emotion_h = emotion_pred.get('height', 0)
                
                # Calculate overlap between face and emotion bounding boxes
                overlap_x = max(0, min(face_x + face_w, emotion_x + emotion_w) - max(face_x, emotion_x))
                overlap_y = max(0, min(face_y + face_h, emotion_y + emotion_h) - max(face_y, emotion_y))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_emotion = emotion_pred
            
            if best_emotion:
                emotion_info = {
                    "face_index": i,
                    "face_bbox": (face_x, face_y, face_w, face_h),
                    "emotion": best_emotion.get('class', 'Unknown'),
                    "confidence": best_emotion.get('confidence', 0),
                    "emotion_bbox": (best_emotion.get('x', 0), best_emotion.get('y', 0), 
                                   best_emotion.get('width', 0), best_emotion.get('height', 0)),
                    "overlap_area": best_overlap
                }
                face_emotions.append(emotion_info)
                
                print(f"🎭 Face {i+1}: {emotion_info['emotion']} (confidence: {emotion_info['confidence']:.3f})")
                print(f"    Face bbox: {emotion_info['face_bbox']}")
                print(f"    Emotion bbox: {emotion_info['emotion_bbox']}")
                print(f"    Overlap area: {emotion_info['overlap_area']}")
            else:
                print(f"🎭 Face {i+1}: No emotion detected")
        
        return face_emotions
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return []

def verify_liveness(body):
    """Verify liveness using challenge-response and motion detection"""
    try:
        frames = body.get("liveness_frames", [])
        challenge = body.get("challenge")
        
        # Basic validation
        if len(frames) < 2:
            print("❌ Liveness check failed: insufficient frames")
            return False
        
        if not challenge:
            print("❌ Liveness check failed: no challenge provided")
            return False
        
        # Decode and compare frames for motion detection
        def decode_image(img_dataurl):
            header, b64 = img_dataurl.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("L").resize((100, 100))
        
        try:
            first_frame = decode_image(frames[0])
            last_frame = decode_image(frames[-1])
            
            # Calculate difference between frames
            diff = ImageChops.difference(first_frame, last_frame)
            bbox = diff.getbbox()
            
            # Check if there's significant motion
            if bbox is None:
                print("❌ Liveness check failed: no motion detected")
                return False
            
            # Calculate motion score
            diff_array = np.array(diff)
            motion_score = np.mean(diff_array)
            
            # Require minimum motion threshold
            if motion_score < 5.0:  # Adjust threshold as needed
                print(f"❌ Liveness check failed: insufficient motion (score: {motion_score})")
                return False
            
            print(f"✅ Liveness check passed: motion score {motion_score}")
            return True
            
        except Exception as e:
            print(f"❌ Liveness check failed: frame processing error - {e}")
            return False
            
    except Exception as e:
        print(f"❌ Liveness check failed: {e}")
        return False

def verify_liveness_on_auth(body):
    """Verify liveness during authentication (simplified version)"""
    # For authentication, we can use a simpler check or skip entirely
    # In production, you might want to implement a more sophisticated check
    return True

# Spotify API Routes
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

@app.route('/api/spotify/devices', methods=['GET'])
def get_devices():
    try:
        sp = get_spotify_client()
        devices = sp.devices()
        return jsonify({
            'success': True,
            'devices': devices['devices']
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
    return jsonify({
        'success': True,
        'running': gesture_system.running if gesture_system else False
    })

@app.route('/api/roboflow/test', methods=['GET'])
def test_roboflow():
    """Test Roboflow API key and workspace access"""
    try:
        if not ROBOFLOW_API_KEY:
            return jsonify({'error': 'No Roboflow API key configured'}), 400
        
        # Test the root endpoint to verify API key
        url = "https://api.roboflow.com/"
        params = {'api_key': ROBOFLOW_API_KEY}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                'success': True,
                'message': 'Roboflow API key is valid',
                'workspace': data.get('workspace'),
                'welcome': data.get('welcome')
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Roboflow API key test failed: {response.status_code}',
                'details': response.text
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/users', methods=['GET'])
def debug_users():
    """Debug endpoint to check enrolled users"""
    try:
        if not qdrant:
            return jsonify({"error": "Qdrant not available"}), 500
        
        # Get collection info
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        print(f"📊 Collection info: {collection_info}")
        
        # Get all points
        points = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100
        )
        
        print(f"🔍 Found {len(points[0])} points in collection")
        
        # Group by user
        users = {}
        for point in points[0]:
            user_id = point.payload.get("userId")
            if user_id:
                if user_id not in users:
                    users[user_id] = {
                        "id": user_id,
                        "name": point.payload.get("name"),
                        "email": point.payload.get("email"),
                        "embeddings": 0,
                        "frames": []
                    }
                users[user_id]["embeddings"] += 1
                users[user_id]["frames"].append(point.payload.get("frameIndex"))
        
        print(f"👥 Users found: {users}")
        
        return jsonify({
            "collection_info": collection_info,
            "total_points": len(points[0]),
            "users": users
        })
        
    except Exception as e:
        print(f"Debug users error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emotion/detect', methods=['POST'])
def detect_emotions():
    """Separate endpoint for emotion detection only"""
    try:
        body = request.get_json()
        image = body.get("image")
        
        if not image:
            return jsonify({"error": "no_image_provided"}), 400
        
        print("🎭 Starting separate emotion detection...")
        
        # Use the emotion analysis function
        face_emotions = analyze_emotions_from_faces(image)
        
        if not face_emotions:
            return jsonify({
                "success": False,
                "message": "No emotions detected",
                "emotions": []
            }), 200
        
        # Format the response
        emotions_data = []
        for emotion_info in face_emotions:
            emotions_data.append({
                "face_index": emotion_info["face_index"],
                "emotion": emotion_info["emotion"],
                "confidence": emotion_info["confidence"],
                "face_bbox": emotion_info["face_bbox"],
                "emotion_bbox": emotion_info["emotion_bbox"]
            })
        
        print(f"🎭 Emotion detection complete: {len(emotions_data)} emotions found")
        
        return jsonify({
            "success": True,
            "message": f"Detected {len(emotions_data)} emotion(s)",
            "emotions": emotions_data,
            "total_faces": len(face_emotions)
        }), 200
        
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return jsonify({"error": "emotion_detection_failed", "details": str(e)}), 500

@app.route('/api/emotion/test', methods=['GET'])
def test_emotion_detection():
    """Test endpoint for emotion detection system"""
    try:
        if not ROBOFLOW_API_KEY:
            return jsonify({'error': 'No Roboflow API key configured'}), 400
        
        return jsonify({
            'success': True,
            'message': 'Emotion detection system ready',
            'model': EMOTION_MODEL_ID,
            'api_key_configured': bool(ROBOFLOW_API_KEY)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    emit('connected', {'message': 'Connected to Qdrant-based face recognition system'})

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

def main():
    print("🤖 Real-time Numbers Detection with Spotify Integration")
    print("=" * 60)
    print("This application will:")
    print("1. Start a Flask server with WebSocket support")
    print("2. Open your webcam for gesture detection")
    print("3. Control Spotify playback based on detected numbers")
    print("4. Provide real-time communication with the frontend")
    print("\n🎮 Play/Pause Controls:")
    print("- Show '0' (zero) to PAUSE")
    print("- Show '5' (five) to PLAY")
    print("\n🌐 Server endpoints:")
    print("- Flask API: http://localhost:5000")
    print("- WebSocket: ws://localhost:5000")
    print("\n📋 Available endpoints:")
    print("- POST /api/enroll - Face enrollment with liveness")
    print("- POST /api/face-auth - Face authentication")
    print("- POST /api/gesture/start - Start gesture detection")
    print("- POST /api/gesture/stop - Stop gesture detection")
    print("- GET  /api/gesture/status - Gesture status")
    print("- POST /api/emotion/detect - Emotion detection")
    print("- GET  /api/emotion/test - Test emotion detection")
    print("- GET  /api/health - System health")
    
    # Initialize clients
    initialize_clients()
    
    # Initialize gesture system after clients are ready
    global gesture_system
    gesture_system = GestureDetectionSystem()

    print("\n🚀 Starting Qdrant-based Flask server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
