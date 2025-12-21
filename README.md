# ReachyMini Face Recognition System

A face recognition system for ReachyMini robot that recognizes faces and calls out names using text-to-speech.

## Features

- ✅ Real-time face recognition from webcam
- ✅ Text-to-speech greeting when recognized faces appear
- ✅ Add new faces from images or camera
- ✅ Visual feedback with bounding boxes and labels
- ✅ Cooldown system to prevent spam greetings
- ✅ Easy to extend with ReachyMini robot control

## Requirements

- Python 3.8+
- Webcam
- macOS, Linux, or Windows

## Quick Clone & Setup

If cloning from Git:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/reachy-face-recognition.git
cd reachy-face-recognition

# Install dependencies
pip install -r requirements.txt
```

Then follow the installation steps below.

## Installation

1. **Install system dependencies** (required for dlib):

   **macOS:**
   ```bash
   brew install cmake
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install cmake
   ```

   **Windows:**
   Download and install CMake from https://cmake.org/download/

2. **Install Python dependencies:**
   ```bash
   cd reachy-face-recognition
   pip install -r requirements.txt
   ```

   Note: Installing `dlib` and `face_recognition` may take several minutes as they compile from source.

## Quick Start

### 1. Add Known Faces

**Option A: From an image file**
```bash
python add_face.py "John Doe" --image path/to/john_photo.jpg
```

**Option B: Capture from camera**
```bash
python add_face.py "Jane Smith" --camera
```

Faces will be saved in the `known_faces/` directory.

### 2. Run Face Recognition

```bash
python face_recognition_system.py
```

The system will:
- Open your webcam
- Detect and recognize faces in real-time
- Greet recognized people by name
- Display bounding boxes and labels

### 3. Controls

- **'q'**: Quit the application
- **'a'**: Add the current face to the database (prompts for name)

## Usage Examples

### Basic Usage
```bash
# Run with default settings (camera 0, tolerance 0.6)
python face_recognition_system.py
```

### Custom Camera
```bash
# Use a different camera (e.g., camera 1)
python face_recognition_system.py --camera 1
```

### Adjust Recognition Sensitivity
```bash
# More strict (fewer false positives, may miss some matches)
python face_recognition_system.py --tolerance 0.5

# More lenient (more matches, may have false positives)
python face_recognition_system.py --tolerance 0.7
```

### Custom Faces Directory
```bash
python face_recognition_system.py --faces-dir my_faces
```

## Project Structure

```
reachy-face-recognition/
├── face_recognition_system.py  # Main face recognition system
├── add_face.py                 # Utility to add faces
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── known_faces/                # Directory for known face images
    ├── John_Doe.jpg
    ├── Jane_Smith.jpg
    └── ...
```

## How It Works

1. **Face Detection**: Uses `face_recognition` library (based on dlib) to detect faces in video frames
2. **Face Encoding**: Converts detected faces into 128-dimensional encodings
3. **Face Matching**: Compares detected face encodings with known face encodings using Euclidean distance
4. **Recognition**: If distance is below tolerance threshold, the face is recognized
5. **Text-to-Speech**: Uses `pyttsx3` to speak the recognized person's name
6. **Cooldown**: Prevents greeting the same person multiple times in quick succession

## Customization

### Adjust Recognition Tolerance

Edit the tolerance value in `face_recognition_system.py`:
```python
system = FaceRecognitionSystem(tolerance=0.6)  # Lower = more strict
```

### Change Speech Rate/Volume

Modify TTS settings in the `_configure_tts()` method:
```python
self.tts_engine.setProperty('rate', 150)      # Words per minute
self.tts_engine.setProperty('volume', 0.9)    # 0.0 to 1.0
```

### Adjust Cooldown Time

Change the recognition cooldown to prevent spam:
```python
self.recognition_cooldown = 3.0  # seconds
```

## Integration with ReachyMini

To integrate with ReachyMini robot control, you can extend the `greet_person()` method:

```python
from reachy_sdk import ReachySDK

# In __init__:
self.reachy = ReachySDK(host='reachy.local')  # Adjust host as needed

# In greet_person():
def greet_person(self, name: str):
    # ... existing code ...
    
    # Optional: Make ReachyMini wave or look at person
    # self.reachy.head.look_at(x, y, z)
    # self.reachy.left_arm.goto_posture('wave')
```

## Troubleshooting

### "No module named 'dlib'"
- Make sure CMake is installed
- Try: `pip install --upgrade dlib`

### "No face found in image"
- Ensure the image contains a clear, front-facing face
- Try images with better lighting
- Face should be clearly visible and not too small

### Camera not working
- Check camera permissions (especially on macOS)
- Try different camera indices: `--camera 1`, `--camera 2`, etc.
- On Linux, you may need to install: `sudo apt-get install v4l-utils`

### Recognition not working well
- Add multiple photos of the same person from different angles
- Ensure good lighting conditions
- Adjust tolerance value (try 0.5-0.7 range)
- Make sure faces are clearly visible in the camera

### Text-to-speech not working
- On Linux, you may need: `sudo apt-get install espeak`
- On macOS, should work out of the box
- On Windows, uses SAPI5 (should work by default)

## Performance Tips

- Process every Nth frame instead of every frame for better performance
- Reduce camera resolution if experiencing lag
- Close other applications using the camera

## License

This project is provided as-is for use with ReachyMini robots.

## Credits

- Uses [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- Built for [ReachyMini](https://www.pollen-robotics.com/reachy/) by Pollen Robotics

