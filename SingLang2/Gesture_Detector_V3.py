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

        # Drawing mode
        self.drawing_mode = False
        self.drawing_points = []
        self.last_index_tip = None
        self.drawing_enabled = True  # Toggle for drawing feature

        # TTS toggle
        self.tts_enabled = True

        # Text display toggle
        self.text_enabled = True

        # HUD panel toggle
        self.hud_enabled = True

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
        if not self.tts_enabled:
            return  # Skip TTS if disabled

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

    def check_drawing_mode(self, landmarks):
        """Check if thumb and index finger are extended for drawing"""
        fingers = self.count_fingers(landmarks)

        # Thumb + index finger up = drawing mode (like a gun/pointer gesture)
        if fingers == [1, 1, 0, 0, 0]:  # Thumb + index only
            return True
        return False

    def check_eraser_mode(self, landmarks):
        """Check if pinky is extended for eraser mode"""
        fingers = self.count_fingers(landmarks)

        # Only pinky up = eraser mode
        if fingers == [0, 0, 0, 0, 1]:  # Only pinky
            return True
        return False

    def draw_air_drawing(self, frame, hand_landmarks):
        """Draw with index finger in the air or erase with pinky"""
        if not self.drawing_enabled:
            return frame  # Skip drawing if disabled

        h, w = frame.shape[:2]

        # Get index finger tip position (landmark 8) for drawing
        index_tip = hand_landmarks[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # Get pinky tip position (landmark 20) for erasing
        pinky_tip = hand_landmarks[20]
        px = int(pinky_tip.x * w)
        py = int(pinky_tip.y * h)

        # Check for eraser mode first (pinky only)
        if self.check_eraser_mode(hand_landmarks):
            self.drawing_mode = False
            self.last_index_tip = None

            # Erase nearby lines
            erase_radius = 30
            self.drawing_points = [
                line for line in self.drawing_points
                if not (
                        min(abs(line[0][0] - px), abs(line[1][0] - px)) < erase_radius and
                        min(abs(line[0][1] - py), abs(line[1][1] - py)) < erase_radius
                )
            ]

            # Draw eraser indicator
            cv2.circle(frame, (px, py), erase_radius, (0, 0, 255), 2)  # Red circle
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(frame, "ERASER", (px + 20, py - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Check for drawing mode
        elif self.check_drawing_mode(hand_landmarks):
            # Drawing mode active
            self.drawing_mode = True

            # Add point to drawing
            if self.last_index_tip is not None:
                self.drawing_points.append(((self.last_index_tip[0], self.last_index_tip[1]), (x, y)))

            self.last_index_tip = (x, y)

            # Draw indicator at fingertip
            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)  # Magenta dot
            cv2.circle(frame, (x, y), 15, (255, 0, 255), 2)  # Outer ring

            # Show "DRAWING MODE" indicator
            cv2.putText(frame, "DRAWING MODE", (x + 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            self.drawing_mode = False
            self.last_index_tip = None

        # Draw all lines
        for line in self.drawing_points:
            cv2.line(frame, line[0], line[1], (255, 0, 255), 3)

        return frame

    def draw_tech_hud(self, frame, hand_detected):
        """Draw tech HUD overlay with stats"""
        if not self.hud_enabled:
            return frame  # Skip HUD if disabled

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

        # Drawing mode indicator
        if self.drawing_mode and self.drawing_enabled:
            cv2.putText(frame, "Air Draw: ENABLED", (w - 240, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # TTS status indicator
        tts_color = (0, 255, 0) if self.tts_enabled else (100, 100, 100)
        tts_text = "ON" if self.tts_enabled else "OFF"
        cv2.putText(frame, f"TTS: {tts_text}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 2)

        # Text display status indicator
        text_color = (0, 255, 0) if self.text_enabled else (100, 100, 100)
        text_status = "ON" if self.text_enabled else "OFF"
        cv2.putText(frame, f"TEXT: {text_status}", (w - 120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

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
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Corner brackets for tech look
        bracket_len = 20
        # Top-left
        cv2.line(frame, (x_min, y_min), (x_min + bracket_len, y_min), (0, 255, 0), 2)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + bracket_len), (0, 255, 0), 2)
        # Top-right
        cv2.line(frame, (x_max, y_min), (x_max - bracket_len, y_min), (0, 255, 0), 2)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + bracket_len), (0, 255, 0), 2)
        # Bottom-left
        cv2.line(frame, (x_min, y_max), (x_min + bracket_len, y_max), (0, 255, 0), 2)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - bracket_len), (0, 255, 0), 2)
        # Bottom-right
        cv2.line(frame, (x_max, y_max), (x_max - bracket_len, y_max), (0, 255, 0), 2)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - bracket_len), (0, 255, 0), 2)

        # Animated scan line
        scan_y = y_min + int((y_max - y_min) * ((self.frame_count % 60) / 60))
        cv2.line(frame, (x_min, scan_y), (x_max, scan_y), (0, 255, 255), 1)

        # Label
        cv2.putText(frame, "hand", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 2)

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
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
        print("\n✨ BONUS FEATURES:")
        print("  👆  Thumb + Index (Gun gesture) = AIR DRAWING MODE")
        print("  🤙  Pinky Only = ERASER MODE")
        print("\n🎮 CONTROLS:")
        print("  Press 'c' = Clear all drawings")
        print("  Press 'd' = Toggle DRAWING feature ON/OFF")
        print("  Press 's' = Toggle TTS (speech) ON/OFF")
        print("  Press 't' = Toggle TEXT display ON/OFF")
        print("  Press 'h' = Toggle HUD panels ON/OFF")
        print("  Press 'q' = Quit")
        print("=" * 50)
        print("\nHold each gesture for 3 seconds!")
        print(
            f"DRAW: {'ON' if self.drawing_enabled else 'OFF'} | TTS: {'ON' if self.tts_enabled else 'OFF'} | TEXT: {'ON' if self.text_enabled else 'OFF'} | HUD: {'ON' if self.hud_enabled else 'OFF'}\n")

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

                # Air drawing feature
                frame = self.draw_air_drawing(frame, hand_landmarks)

                # Draw hand skeleton
                frame = self.draw_landmarks(frame, hand_landmarks)

                # Recognize gesture (skip if in drawing mode)
                if not self.drawing_mode and self.drawing_enabled:
                    current_gesture = self.recognize_gesture(hand_landmarks)
                elif not self.drawing_enabled:
                    # Always recognize gestures if drawing is disabled
                    current_gesture = self.recognize_gesture(hand_landmarks)
            else:
                self.detection_confidence = 0.0

            # Draw tech HUD overlay
            frame = self.draw_tech_hud(frame, hand_detected)

            # Display gesture and speak
            if current_gesture:
                # Split long text into two lines (only if text is enabled)
                if self.text_enabled:
                    if current_gesture == "Tech Nova Three Point Zero":
                        cv2.putText(frame, "Tech Nova", (20, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.putText(frame, "Three Point Zero", (30, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    elif current_gesture == "Welcome to Tech Nova":
                        cv2.putText(frame, "Welcome to", (50, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.putText(frame, "Tech Nova", (50, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    else:
                        cv2.putText(frame, current_gesture, (20, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # Speak if gesture changed
                current_time = time.time()
                if (current_gesture != self.last_gesture and
                        current_time - self.last_speak_time > self.speak_cooldown):
                    print(f"Speaking: {current_gesture}")
                    self.speak(current_gesture)
                    self.last_speak_time = current_time
                    self.last_gesture = current_gesture
            else:
                if self.text_enabled:
                    cv2.putText(frame, "Show a gesture", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Sign Language Detection', frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear drawing
                self.drawing_points = []
                print("Drawing cleared!")
            elif key == ord('d'):
                # Toggle drawing feature
                self.drawing_enabled = not self.drawing_enabled
                if not self.drawing_enabled:
                    self.drawing_points = []  # Clear drawings when disabled
                status = "ON" if self.drawing_enabled else "OFF"
                print(f"DRAWING feature {status}")
            elif key == ord('s'):
                # Toggle TTS
                self.tts_enabled = not self.tts_enabled
                status = "ON" if self.tts_enabled else "OFF"
                print(f"TTS {status}")
            elif key == ord('t'):
                # Toggle text display
                self.text_enabled = not self.text_enabled
                status = "ON" if self.text_enabled else "OFF"
                print(f"TEXT display {status}")
            elif key == ord('h'):
                # Toggle HUD panels
                self.hud_enabled = not self.hud_enabled
                status = "ON" if self.hud_enabled else "OFF"
                print(f"HUD panels {status}")

        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run()