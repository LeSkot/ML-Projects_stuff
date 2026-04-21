import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
from gtts import gTTS
import os
import tempfile
import pygame

class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Gesture tracking
        self.last_gesture = None
        self.last_speak_time = 0
        self.speak_cooldown = 1  # Faster cooldown
        
        # Pre-generate audio files for instant playback
        self.audio_cache = {}
        self.preload_audio()
        
    def preload_audio(self):
        """Pre-generate and preload all audio files into memory"""
        phrases = [
            "Welcome to Tech Nova",
            "Three Point Zero",
            "Let's Get Started",
            "Are You Ready",
            "Let's Begin",
            "Tech Nova Three Point Zero"
        ]
        
        print("Preloading audio files...")
        for phrase in phrases:
            filename = f"audio_{phrase.replace(' ', '_')}.mp3"
            
            if not os.path.exists(filename):
                print(f"  Generating: {phrase}")
                tts = gTTS(text=phrase, lang='en', slow=False)
                tts.save(filename)
            
            # Load sound into memory for instant playback
            self.audio_cache[phrase] = pygame.mixer.Sound(filename)
        
        print("Audio preloaded into memory! ✓\n")
    
    def count_fingers(self, landmarks):
        """Count extended fingers"""
        fingers = []
        
        # Thumb (check x-coordinate)
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers - check if tip is above middle joint
        finger_tips = [8, 12, 16, 20]
        finger_mids = [6, 10, 14, 18]
        
        for tip, mid in zip(finger_tips, finger_mids):
            if landmarks[tip].y < landmarks[mid].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def recognize_gesture(self, landmarks):
        """Recognize gesture based on finger count"""
        fingers = self.count_fingers(landmarks)
        finger_count = sum(fingers)
        
        # Tech Nova 3.0 Assembly Gestures (FULL PHRASES):
        
        # Open palm (5 fingers) = "Welcome to Tech Nova"
        if finger_count == 5:
            return "Welcome to Tech Nova"
        
        # 4 fingers = "Three Point Zero"
        elif finger_count == 4 and fingers[0] == 0:
            return "Three Point Zero"
        
        # 3 fingers = "Let's Get Started"
        elif finger_count == 3 and fingers[0] == 0:
            return "Let's Get Started"
        
        # Peace sign (2 fingers) = "Are You Ready"
        elif finger_count == 2 and fingers[1] == 1 and fingers[2] == 1:
            return "Are You Ready"
        
        # Thumbs up = "Let's Begin"
        elif finger_count == 1 and fingers[0] == 1:
            return "Let's Begin"
        
        # Closed fist (0 fingers) = "Tech Nova Three Point Zero"
        elif finger_count == 0:
            return "Tech Nova Three Point Zero"
        
        return None
    
    def speak(self, text):
        """Speak text without blocking video - INSTANT!"""
        try:
            print(f">>> SPEAKING: '{text}'")
            
            # Use pre-loaded Sound object from memory
            if text in self.audio_cache:
                sound = self.audio_cache[text]
                sound.play()  # Play and DON'T wait - keeps video smooth!
            else:
                # Fallback: generate on the fly if not cached
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    audio_file = fp.name
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(audio_file)
                
                sound = pygame.mixer.Sound(audio_file)
                sound.play()
            
            # NO WAITING - let it play in background while video continues!
            
        except Exception as e:
            print(f"!!! TTS ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks manually"""
        h, w, c = frame.shape
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_point, end_point, (255, 255, 0), 2)
        
        # Draw landmark points
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 0, 127), -1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        # Lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting Sign Language Detection...")
        print("TECH NOVA 3.0 - ASSEMBLY DEMO")
        print("=" * 50)
        print("Gesture Sequence:")
        print("  🖐️  5 Fingers = 'Welcome to Tech Nova'")
        print("  🖖  4 Fingers = 'Three Point Zero'")
        print("  🤟  3 Fingers = 'Let's Get Started'")
        print("  ✌️  2 Fingers = 'Are You Ready'")
        print("  👍  Thumbs Up = 'Let's Begin'")
        print("  ✊  Fist = 'Tech Nova Three Point Zero'")
        print("=" * 50)
        print("\nHold each gesture for 3 seconds!")
        print("Press 'q' to quit\n")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            detection_result = self.detector.detect(mp_image)
            
            current_gesture = None
            
            # Process results
            if detection_result.hand_landmarks:
                # Draw landmarks
                hand_landmarks = detection_result.hand_landmarks[0]
                frame = self.draw_landmarks(frame, hand_landmarks)
                
                # Recognize gesture
                current_gesture = self.recognize_gesture(hand_landmarks)
            
            # Display gesture and speak
            if current_gesture:
                # Split long text into two lines
                if current_gesture == "Tech Nova Three Point Zero":
                    cv2.putText(frame, "Tech Nova", (20, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, "Three Point Zero", (20, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, current_gesture, (20, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Speak if gesture changed
                current_time = time.time()
                if (current_gesture != self.last_gesture and 
                    current_time - self.last_speak_time > self.speak_cooldown):
                    print(f"Speaking: {current_gesture}")
                    self.speak(current_gesture)
                    self.last_speak_time = current_time
                    self.last_gesture = current_gesture
            else:
                cv2.putText(frame, "Show a gesture", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run()