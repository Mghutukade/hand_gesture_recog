# Hand Gesture Recognition (MediaPipe + SVM)

A real-time hand gesture recognition system that detects a single hand, extracts 3D landmarks using MediaPipe Hands, and classifies user-defined gestures with an SVM. Designed for intuitive HCI and gesture-based control.

## Features
- Real-time webcam inference (30–60 FPS on CPU)
- Simple data collection loop (label-by-label)
- Reproducible ML pipeline (scaler + SVM, GridSearchCV)
- Clean project structure; easy to extend

## Gestures
Start with: palm, fist, thumbs_up, ok, peace, stop (customize as needed).

## Setup
```bash
git clone <your-repo-url>
cd hand-gesture-recognition
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
