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


class FaceRecognitionSystem:
    """Main face recognition system for ReachyMini"""
    
    def __init__(self, known_faces_dir: str = "known_faces", tolerance: float = 0.6):
        """
        Initialize the face recognition system
        
        Args:
            known_faces_dir: Directory containing known face images
            tolerance: How much distance between faces to consider it a match (lower = more strict)
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
        
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
    
    def speak(self, text: str):
        """Speak text using text-to-speech"""
        print(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
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
    
    args = parser.parse_args()
    
    # Create and run the system
    system = FaceRecognitionSystem(
        known_faces_dir=args.faces_dir,
        tolerance=args.tolerance
    )
    
    system.run(camera_index=args.camera)


if __name__ == "__main__":
    main()

