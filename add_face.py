#!/usr/bin/env python3
"""
Utility script to add faces to the known faces database
"""

import cv2
import face_recognition
import argparse
from pathlib import Path
import shutil


def add_face_from_image(image_path: str, name: str, known_faces_dir: str = "known_faces"):
    """
    Add a face from an image file to the known faces database
    
    Args:
        image_path: Path to the image file
        name: Name of the person
        known_faces_dir: Directory to save known faces
    """
    known_faces_path = Path(known_faces_dir)
    known_faces_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = face_recognition.load_image_file(image_path)
    
    # Find faces
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        print("Error: No face found in the image")
        return False
    
    if len(face_locations) > 1:
        print(f"Warning: Multiple faces found ({len(face_locations)}). Using the first one.")
    
    # Get encoding for the first face
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    
    # Save the image to known_faces directory
    save_path = known_faces_path / f"{name}.jpg"
    shutil.copy(image_path, save_path)
    
    print(f"✓ Added face: {name}")
    print(f"  Saved to: {save_path}")
    return True


def add_face_from_camera(name: str, known_faces_dir: str = "known_faces", camera_index: int = 0):
    """
    Capture a face from camera and add to database
    
    Args:
        name: Name of the person
        known_faces_dir: Directory to save known faces
        camera_index: Camera device index
    """
    print(f"Capturing face for: {name}")
    print("Press SPACE to capture, ESC to cancel")
    
    video_capture = cv2.VideoCapture(camera_index)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return False
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw rectangle around face
            if face_locations:
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Capture Face - Press SPACE to capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar
                if face_locations:
                    # Save the frame
                    temp_path = f"temp_capture_{name}.jpg"
                    cv2.imwrite(temp_path, frame)
                    success = add_face_from_image(temp_path, name, known_faces_dir)
                    Path(temp_path).unlink()  # Delete temp file
                    if success:
                        print("Face captured successfully!")
                    break
                else:
                    print("No face detected. Please position yourself in front of the camera.")
            elif key == 27:  # ESC
                print("Cancelled")
                break
                
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Add a face to the known faces database')
    parser.add_argument('name', help='Name of the person')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--camera', action='store_true', help='Capture from camera')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--faces-dir', type=str, default='known_faces',
                       help='Directory for known faces (default: known_faces)')
    
    args = parser.parse_args()
    
    if args.image:
        add_face_from_image(args.image, args.name, args.faces_dir)
    elif args.camera:
        add_face_from_camera(args.name, args.faces_dir, args.camera_index)
    else:
        print("Error: Please specify --image or --camera")
        parser.print_help()


if __name__ == "__main__":
    main()

