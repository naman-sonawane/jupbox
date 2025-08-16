import cv2
import numpy as np
import requests
import json
import time
from PIL import Image
import io
import base64
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

API_KEY = "t4y8okUvSiM9Y9QdOhia"
MODEL_ID = "numbers-qysva/7"

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

SPOTIFY_CLIENT_ID = "b52e7240c7544a589e65126efac853dc"
SPOTIFY_CLIENT_SECRET = "d54a64dfe1e144c4b2e0fa3cb256ad53"
REDIRECT_URI = "http://127.0.0.1:8888/callback"

sp = None

def get_spotify_client():
    global sp
    if sp is None:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing user-read-private"
        ))
    return sp

class WebcamNumbersDetector:
    def __init__(self):
        self.api_key = API_KEY
        self.model_id = MODEL_ID
        self.cap = None
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 1.0
        self.is_playing = True
        self.last_toggle_time = 0
        self.toggle_cooldown = 2.0
        self.running = False
        
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        print("Webcam started successfully!")
        return True
    
    def encode_image_to_base64(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb_frame)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def detect_numbers_api(self, frame):
        try:
            temp_filename = f"temp_frame_{int(time.time())}.jpg"
            cv2.imwrite(temp_filename, frame)
            
            from inference_sdk import InferenceHTTPClient
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
            
            result = client.infer(temp_filename, model_id=self.model_id)
            
            import os
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            return result
            
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def draw_predictions(self, frame, predictions):
        if not predictions:
            return frame
        
        for pred in predictions:
            bbox = pred.get('bbox', {})
            if not bbox:
                continue
            
            x = int(bbox.get('x', 0))
            y = int(bbox.get('y', 0))
            width = int(bbox.get('width', 0))
            height = int(bbox.get('height', 0))
            
            x1 = x - width // 2
            y1 = y - height // 2
            x2 = x + width // 2
            y2 = y + height // 2
            
            number_class = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', 0)
            color = pred.get('color', '#8622FF')
            
            color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            label = f"{number_class}: {confidence:.2%}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width + 10, y1), color_bgr, -1)
            
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def handle_play_pause_toggle(self, predictions):
        current_time = time.time()
        
        if current_time - self.last_toggle_time < self.toggle_cooldown:
            return
        
        for pred in predictions:
            number_class = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', 0)
            
            if confidence > 0.7:
                if number_class == "0" and self.is_playing:
                    self.is_playing = False
                    self.last_toggle_time = current_time
                    print(f"🎵 PAUSED - Detected class '0' with {confidence:.1%} confidence")
                    
                    self.control_spotify_playback(False)
                    
                    socketio.emit('playback_state_changed', {'is_playing': False})
                    
                elif number_class == "5" and not self.is_playing:
                    self.is_playing = True
                    self.last_toggle_time = current_time
                    print(f"> PLAYING - Detected class '5' with {confidence:.1%} confidence")
                    
                    self.control_spotify_playback(True)
                    
                    socketio.emit('playback_state_changed', {'is_playing': True})

    def control_spotify_playback(self, should_play):
        try:
            spotify_client = get_spotify_client()
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

detector = None
detector_thread = None

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
                'image': user['images'][0]['url'] if user['images'] else None
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/search', methods=['GET'])
def search_tracks():
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter required'}), 400
        
        sp = get_spotify_client()
        results = sp.search(q=query, type='track', limit=10)
        
        tracks = []
        for item in results['tracks']['items']:
            tracks.append({
                'name': item['name'],
                'artist': item['artists'][0]['name'],
                'album': item['album']['name'],
                'cover_url': item['album']['images'][0]['url'] if item['album']['images'] else None,
                'uri': item['uri']
            })
        
        return jsonify({'success': True, 'tracks': tracks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/play-track', methods=['POST'])
def play_specific_track():
    try:
        data = request.get_json()
        uri = data.get('uri')
        
        if not uri:
            return jsonify({'success': False, 'error': 'URI required'}), 400
        
        sp = get_spotify_client()
        sp.start_playback(uris=[uri])
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global detector, detector_thread
    
    try:
        if detector is None:
            detector = WebcamNumbersDetector()
            detector_thread = threading.Thread(target=detector.run)
            detector_thread.daemon = True
            detector_thread.start()
            return jsonify({'success': True, 'message': 'Camera started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global detector
    
    try:
        if detector:
            detector.stop()
            detector = None
            return jsonify({'success': True, 'message': 'Camera stopped successfully'})
        else:
            return jsonify({'success': False, 'message': 'Camera not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    global detector
    return jsonify({
        'success': True,
        'running': detector is not None and detector.running
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')
    emit('connected', {'message': 'Connected to camera control system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from WebSocket')

@socketio.on('request_playback_state')
def handle_playback_state_request():
    try:
        sp = get_spotify_client()
        current = sp.current_playback()
        is_playing = current['is_playing'] if current else False
        emit('playback_state', {'is_playing': is_playing})
    except Exception as e:
        emit('playback_state', {'is_playing': False, 'error': str(e)})

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
    
    global detector, detector_thread
    detector = WebcamNumbersDetector()
    detector_thread = threading.Thread(target=detector.run)
    detector_thread.daemon = True
    detector_thread.start()
    
    print("\n🚀 Starting Flask server with WebSocket support...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()