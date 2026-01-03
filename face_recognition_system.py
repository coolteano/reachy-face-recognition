#!/usr/bin/env python3
"""
ReachyMini Face Recognition System
Recognizes faces and calls out names using text-to-speech
"""

import cv2
import face_recognition
import numpy as np
import os
import json
import pyttsx3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import tempfile
import soundfile as sf

# gTTS for generating audio files for robot
try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# ReachyMini SDK for robot speaker
try:
    from reachy_mini import ReachyMini
    REACHY_AVAILABLE = True
except ImportError:
    REACHY_AVAILABLE = False
    print("Warning: reachy-mini not installed. Audio will play from laptop.")


class FaceRecognitionSystem:
    """Main face recognition system for ReachyMini"""
    
    def __init__(self, known_faces_dir: str = "known_faces", tolerance: float = 0.6, 
                 robot_host: Optional[str] = None, use_robot_speaker: bool = True):
        """
        Initialize the face recognition system
        
        Args:
            known_faces_dir: Directory containing known face images
            tolerance: How much distance between faces to consider it a match (lower = more strict)
            robot_host: ReachyMini robot host/IP (e.g., 'reachy.local' or IP address). 
                       If None, will try to auto-connect.
            use_robot_speaker: If True, use robot's speaker. If False or robot unavailable, use laptop speaker.
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        
        # Initialize ReachyMini robot connection (if available)
        self.robot = None
        self.use_robot_speaker = use_robot_speaker and REACHY_AVAILABLE
        
        if self.use_robot_speaker:
            try:
                print("Connecting to ReachyMini robot...")
                self.robot = ReachyMini(host=robot_host) if robot_host else ReachyMini()
                print("✓ Connected to ReachyMini - audio will play from robot speaker")
            except Exception as e:
                print(f"⚠ Could not connect to ReachyMini: {e}")
                print("   Falling back to laptop speaker")
                self.use_robot_speaker = False
                self.robot = None
        
        # Initialize laptop text-to-speech engine (fallback)
        if not self.use_robot_speaker:
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
        else:
            self.tts_engine = None
        
        # Track recently recognized faces to avoid spam
        self.last_recognition_time: Dict[str, float] = {}
        self.recognition_cooldown = 3.0  # seconds between same person recognition
        
        # Load known faces
        self.load_known_faces()
    
    def _configure_tts(self):
        """Configure text-to-speech settings"""
        # Set speech rate (words per minute)
        self.tts_engine.setProperty('rate', 150)
        
        # Set volume (0.0 to 1.0)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Try to set a more natural voice (if available)
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Prefer female voice if available, otherwise use default
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
    
    def load_known_faces(self):
        """Load all known faces from the known_faces directory"""
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {self.known_faces_dir}")
            print("Add face images (jpg, png) named as 'PersonName.jpg' to this directory")
            return
        
        print(f"Loading known faces from {self.known_faces_dir}...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for image_path in self.known_faces_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                name = image_path.stem  # Get name without extension
                try:
                    # Load image
                    image = face_recognition.load_image_file(str(image_path))
                    # Find face encodings
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        # Use the first face found
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        print(f"  ✓ Loaded: {name}")
                    else:
                        print(f"  ✗ No face found in: {image_path.name}")
                except Exception as e:
                    print(f"  ✗ Error loading {image_path.name}: {e}")
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def add_face(self, image_path: str, name: str) -> bool:
        """
        Add a new face to the known faces database
        
        Args:
            image_path: Path to the image file
            name: Name of the person
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                print(f"No face found in {image_path}")
                return False
            
            # Add to known faces
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            
            # Save a copy to known_faces directory
            save_path = self.known_faces_dir / f"{name}.jpg"
            import shutil
            shutil.copy(image_path, save_path)
            
            print(f"Added face: {name}")
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def recognize_face(self, face_encoding: np.ndarray) -> Optional[str]:
        """
        Recognize a face from encoding
        
        Args:
            face_encoding: Face encoding to match
            
        Returns:
            Name if recognized, None otherwise
        """
        if not self.known_face_encodings:
            return None
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] <= self.tolerance:
            return self.known_face_names[best_match_index]
        
        return None
    
    # def speak(self, text: str):
    #     """Speak text using robot speaker or laptop TTS"""
    #     print(f"Speaking: {text}")
        
    #     if self.use_robot_speaker and self.robot:
    #         try:
    #             # Method 1: Try robot.say() (direct method)
    #             if hasattr(self.robot, 'say'):
    #                 try:
    #                     self.robot.say(text)
    #                     return
    #                 except Exception:
    #                     pass
                
    #             # Method 2: Try robot.speaker.say()
    #             if hasattr(self.robot, 'speaker') and hasattr(self.robot.speaker, 'say'):
    #                 try:
    #                     self.robot.speaker.say(text)
    #                     return
    #                 except Exception:
    #                     pass
                
    #             # Method 3: Generate audio file and play on robot
    #             # Generate audio file, then upload and play on robot
    #             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    #             temp_path = temp_file.name
    #             temp_file.close()
                
    #             try:
    #                 # Generate audio file using gTTS (more reliable than pyttsx3 save_to_file)
    #                 if GTTS_AVAILABLE:
    #                     tts = gTTS(text=text, lang='en', slow=False)
    #                     tts.save(temp_path)
    #                 else:
    #                     # Fallback: use pyttsx3 if gTTS not available
    #                     if self.tts_engine:
    #                         self.tts_engine.save_to_file(text, temp_path)
    #                         self.tts_engine.runAndWait()
    #                     else:
    #                         raise RuntimeError("No TTS engine available")
                    
    #                 # Try to play on robot using speaker.play_sound()
    #                 if hasattr(self.robot, 'speaker') and hasattr(self.robot.speaker, 'play_sound'):
    #                     self.robot.speaker.play_sound(temp_path)
    #                 # Or try uploading and playing via audio API
    #                 elif hasattr(self.robot, 'audio'):
    #                     # Upload and play audio file
    #                     robot_filename = os.path.basename(temp_path)
    #                     self.robot.audio.upload_audio_file(temp_path)
    #                     self.robot.audio.play_audio_file(robot_filename)
    #                 else:
    #                     raise AttributeError("No audio playback method available")
                    
    #                 # Clean up temp file after a delay
    #                 time.sleep(0.5)  # Give time for file to be read
    #                 if os.path.exists(temp_path):
    #                     os.remove(temp_path)
    #                 return
    #             except Exception as e:
    #                 # Clean up temp file on error
    #                 if os.path.exists(temp_path):
    #                     os.remove(temp_path)
    #                 raise e
                
    #             # If all methods fail
    #             raise AttributeError("Robot does not have a valid speak method")
                
    #         except Exception as e:
    #             print(f"Error using robot speaker: {e}")
    #             print("   Falling back to laptop speaker")
    #             # Fallback to laptop speaker
    #             if self.tts_engine:
    #                 self.tts_engine.say(text)
    #                 self.tts_engine.runAndWait()
    #     else:
    #         # Use laptop speaker
    #         if self.tts_engine:
    #             self.tts_engine.say(text)
    #             self.tts_engine.runAndWait()
    #         else:
    #             print(f"⚠ No TTS available. Would say: {text}")
    
   
    def speak(self, text: str):
        """Speak text using ReachyMini's play_sound method"""
        print(f"Speaking: {text}")
        
        # 1. Create the temporary file path
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()

        try:
            # 2. Generate the audio content
            if GTTS_AVAILABLE:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_path)
            else:
                # Fallback to pyttsx3 save if gTTS is missing
                self.tts_engine.save_to_file(text, temp_path)
                self.tts_engine.runAndWait()

            # 3. Play the file using the robot's media interface
            if self.use_robot_speaker and self.robot:
                print(f"Playing {temp_path} on robot...")
                # Using the specific play_sound method from the reachy-mini API
                self.robot.media.play_sound(temp_path)
            else:
                # Local laptop playback fallback
                import subprocess
                # Simple cross-platform way to play mp3 on laptop
                if os.name == 'posix':  # macOS/Linux
                    subprocess.run(['afplay' if 'darwin' in os.sys.platform else 'mpg123', temp_path])
                else:  # Windows
                    os.startfile(temp_path)

        except Exception as e:
            print(f"Error in playback: {e}")
        finally:
            # 4. Cleanup: Remove the file after playing
            # Adding a small sleep ensures the robot has finished reading the file
            time.sleep(1.0)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def greet_person(self, name: str):
        """Greet a recognized person (with cooldown to avoid spam)"""
        current_time = time.time()
        last_time = self.last_recognition_time.get(name, 0)
        
        # Check cooldown
        if current_time - last_time < self.recognition_cooldown:
            return
        
        # Update last recognition time
        self.last_recognition_time[name] = current_time
        
        # Greet the person
        greeting = f"Hello {name}!"
        self.speak(greeting)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Process a video frame for face recognition
        
        Args:
            frame: Video frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (annotated_frame, recognized_names)
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_names = []
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Recognize face
            name = self.recognize_face(face_encoding)
            
            if name:
                recognized_names.append(name)
                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Greet the person
                self.greet_person(name)
            else:
                # Unknown face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "Unknown", (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, recognized_names
    
    def run(self, camera_index: int = 0):
        """
        Run the face recognition system with webcam
        
        Args:
            camera_index: Camera device index (default: 0)
        """
        print("Starting face recognition system...")
        print("Press 'q' to quit")
        print("Press 'a' to add current face to database")
        
        # Initialize camera
        video_capture = cv2.VideoCapture(camera_index)
        
        if not video_capture.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera resolution (optional, for better performance)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        try:
            while True:
                # Grab a single frame
                ret, frame = video_capture.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process every 2nd frame for better performance
                if frame_count % 2 == 0:
                    frame, recognized_names = self.process_frame(frame)
                else:
                    # Still draw rectangles for visual feedback
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                
                # Display the resulting frame
                cv2.imshow('ReachyMini Face Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    # Add current face
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    if face_locations:
                        name = input("Enter name for this face: ").strip()
                        if name:
                            # Extract face region and save
                            top, right, bottom, left = face_locations[0]
                            face_image = rgb_frame[top:bottom, left:right]
                            temp_path = f"temp_face_{name}.jpg"
                            cv2.imwrite(temp_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                            self.add_face(temp_path, name)
                            os.remove(temp_path)
                    else:
                        print("No face detected in current frame")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Clean up
            video_capture.release()
            cv2.destroyAllWindows()
            
            # Close robot connection if open
            if self.robot:
                try:
                    # ReachyMini context manager handles cleanup, but explicit close if needed
                    if hasattr(self.robot, 'close'):
                        self.robot.close()
                except:
                    pass
            
            print("Face recognition system stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ReachyMini Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--tolerance', type=float, default=0.6, 
                       help='Face recognition tolerance (lower = more strict, default: 0.6)')
    parser.add_argument('--faces-dir', type=str, default='known_faces',
                       help='Directory containing known faces (default: known_faces)')
    parser.add_argument('--robot-host', type=str, default=None,
                       help='ReachyMini robot host/IP (e.g., reachy.local or IP address). Auto-detects if not specified.')
    parser.add_argument('--laptop-speaker', action='store_true',
                       help='Force use of laptop speaker instead of robot speaker')
    
    args = parser.parse_args()
    
    # Create and run the system
    system = FaceRecognitionSystem(
        known_faces_dir=args.faces_dir,
        tolerance=args.tolerance,
        robot_host=args.robot_host,
        use_robot_speaker=not args.laptop_speaker
    )
    
    system.run(camera_index=args.camera)


if __name__ == "__main__":
    main()

