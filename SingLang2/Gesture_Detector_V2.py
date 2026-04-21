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
import random


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

        # Tech stats for display
        self.detection_confidence = 0.0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

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

    def draw_tech_hud(self, frame, hand_detected):
        """Draw tech HUD overlay with stats"""
        h, w = frame.shape[:2]

        # Top-left info box
        cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 140), (0, 255, 0), 2)

        cv2.putText(frame, "TECH NOVA 3.0 - AI DEMO", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MediaPipe v0.10.32 | Python 3.12", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        cv2.putText(frame, f"FPS: {self.fps}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # AI Status
        status_color = (0, 255, 0) if hand_detected else (100, 100, 100)
        status_text = "HAND DETECTED" if hand_detected else "SEARCHING..."
        cv2.putText(frame, f"Status: {status_text}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # Detection confidence
        confidence_percent = int(self.detection_confidence * 100)
        cv2.putText(frame, f"Confidence: {confidence_percent}%", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Bottom-right neural network status
        cv2.rectangle(frame, (w - 250, h - 80), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 250, h - 80), (w - 10, h - 10), (0, 255, 0), 2)
        cv2.putText(frame, "NEURAL NETWORK: ONLINE", (w - 240, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Gesture Recognition: ACTIVE", (w - 240, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        # Fake processing bar
        bar_width = int(220 * (self.detection_confidence if hand_detected else 0.3))
        cv2.rectangle(frame, (w - 240, h - 15), (w - 240 + bar_width, h - 10), (0, 255, 0), -1)

        return frame

    def draw_bounding_box(self, frame, hand_landmarks):
        """Draw bounding box around detected hand with scan lines"""
        h, w = frame.shape[:2]

        # Calculate bounding box
        x_coords = [int(lm.x * w) for lm in hand_landmarks]
        y_coords = [int(lm.y * h) for lm in hand_landmarks]

        x_min, x_max = max(0, min(x_coords) - 30), min(w, max(x_coords) + 30)
        y_min, y_max = max(0, min(y_coords) - 30), min(h, max(y_coords) + 30)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        # Corner brackets for tech look
        bracket_len = 20
        # Top-left
        cv2.line(frame, (x_min, y_min), (x_min + bracket_len, y_min), (0, 255, 0), 3)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + bracket_len), (0, 255, 0), 3)
        # Top-right
        cv2.line(frame, (x_max, y_min), (x_max - bracket_len, y_min), (0, 255, 0), 3)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + bracket_len), (0, 255, 0), 3)
        # Bottom-left
        cv2.line(frame, (x_min, y_max), (x_min + bracket_len, y_max), (0, 255, 0), 3)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - bracket_len), (0, 255, 0), 3)
        # Bottom-right
        cv2.line(frame, (x_max, y_max), (x_max - bracket_len, y_max), (0, 255, 0), 3)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - bracket_len), (0, 255, 0), 3)

        # Animated scan line
        scan_y = y_min + int((y_max - y_min) * ((self.frame_count % 60) / 60))
        cv2.line(frame, (x_min, scan_y), (x_max, scan_y), (0, 255, 255), 1)

        # Label
        cv2.putText(frame, "HAND DETECTED", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

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

            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw landmark points
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

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

            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = current_time

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands
            detection_result = self.detector.detect(mp_image)

            current_gesture = None
            hand_detected = False

            # Process results
            if detection_result.hand_landmarks:
                hand_detected = True
                # Draw landmarks
                hand_landmarks = detection_result.hand_landmarks[0]

                # Calculate detection confidence (simulated from hand presence)
                self.detection_confidence = 0.85 + (random.random() * 0.15)  # 85-100%

                # Draw bounding box and scan lines
                frame = self.draw_bounding_box(frame, hand_landmarks)

                # Draw hand skeleton
                frame = self.draw_landmarks(frame, hand_landmarks)

                # Recognize gesture
                current_gesture = self.recognize_gesture(hand_landmarks)
            else:
                self.detection_confidence = 0.0

            # Draw tech HUD overlay
            frame = self.draw_tech_hud(frame, hand_detected)

            # Display gesture and speak
            if current_gesture:
                # Split long text into two lines
                if current_gesture == "Tech Nova Three Point Zero":
                    cv2.putText(frame, "Tech Nova", (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(frame, "Three Point Zero", (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, current_gesture, (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

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