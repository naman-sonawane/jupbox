"""
Real-time Hand Gesture Detection using Webcam and Roboflow API
This script captures video from your webcam and detects hand gestures in real-time.
"""

import cv2
import numpy as np
import requests
import json
import time
from PIL import Image
import io
import base64


API_KEY = "t4y8okUvSiM9Y9QdOhia"
MODEL_ID = "numbers-qysva/7"

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
        
    def start_webcam(self):
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        print("Webcam started successfully!")
        print("Press 'q' to quit, 's' to save current frame")
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
            
            if confidence > 0.15:  # Lowered from 0.4 to 0.15 for better sensitivity
                if number_class == "0" and self.is_playing:
                    
                    self.is_playing = False
                    self.last_toggle_time = current_time
                    print(f"🎵 PAUSED - Detected class '0' with {confidence:.1%} confidence")
                    
                elif number_class == "5" and not self.is_playing:
                                         
                     self.is_playing = True
                     self.last_toggle_time = current_time
                     print(f"> PLAYING - Detected class '5' with {confidence:.1%} confidence")

    def run(self):
        
        if not self.start_webcam():
            return
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                
                frame = cv2.flip(frame, 1)
                
                
                self.frame_count += 1
                cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                
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
    
    def cleanup(self):
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped and resources cleaned up.")

def main():
    
    print("🤖 Real-time Numbers Detection with Play/Pause Control")
    print("=" * 50)
    print("This application will:")
    print("1. Open your webcam")
    print("2. Capture frames in real-time")
    print("3. Send frames to Roboflow API for numbers detection")
    print("4. Display results with bounding boxes")
    print("5. Control play/pause status based on detected numbers")
    print("\n🎮 Play/Pause Controls:")
    print("- Show '0' (zero) to PAUSE")
    print("- Show '5' (five) to PLAY")
    print("\n⌨️  Other Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("\nNote: API calls are limited to once per second to avoid rate limiting")
    
    input("\nPress Enter to start...")
    
    detector = WebcamNumbersDetector()
    detector.run()

if __name__ == "__main__":
    main()
