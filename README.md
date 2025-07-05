# IntelliMotion: Real-Time Hand Gesture and Pose Tracking Using OpenCV and MediaPipe

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Made with OpenCV](https://img.shields.io/badge/Made%20with-OpenCV-brightgreen.svg)](https://opencv.org/)
[![Powered by MediaPipe](https://img.shields.io/badge/Powered%20by-MediaPipe-orange.svg)](https://mediapipe.dev/)

IntelliMotion is a Python-based application for real-time intelligent motion analysis using Google's MediaPipe and OpenCV. It detects and tracks human body landmarks (pose, hands, face) for gesture recognition, pose estimation, and interactive applications. The modular design allows for easy extension and customization.

---

## Table of Contents

* [Demo](#demo)
* [Key Features](#key-features)
* [Technologies Used](#technologies-used)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Running IntelliMotion](#running-intellimotion)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [License](#license)

---
## Demo

<!-- Replace the below with your own GIF or video demo if available -->
<p align="center">
  <img src="Hand Landmarks.png" alt="Hand Landmarks Example" width="350"/>
  <br/>
  <em>Hand Landmarks Example: MediaPipe detects and annotates key points on the hand in real time.</em>
</p>

<p align="center">
  <img src="Pose Landmarks.png" alt="Pose Landmarks Example" width="350"/>
  <br/>
  <em>Pose Landmarks Example: Full-body pose estimation with skeletal connections visualized.</em>
</p>

---

### Gesture & System States Gallery

<!-- Stretched Arm States -->
<p align="center">
  <img src="Interface Images/Full_Stretched_Arm.png" alt="Full Stretched Arm" width="220"/>
  <img src="Interface Images/Partial_Stretched_Arm.png" alt="Partial Stretched Arm" width="220"/>
  <br/>
  <em>Left: Full Stretched Arm - Detected pose with arm fully extended.<br>Right: Partial Stretched Arm - Detected pose with arm partially extended.</em>
</p>

<!-- Volume Gestures -->
<p align="center">
  <img src="Interface Images/Full_Vol_Gesture.png" alt="Full Volume Gesture" width="180"/>
  <img src="Interface Images/Intermediate_Vol_Gesture.png" alt="Intermediate Volume Gesture" width="180"/>
  <img src="Interface Images/Zero_Vol_Gesture.png" alt="Zero Volume Gesture" width="180"/>
  <br/>
  <em>Volume Gestures: Full, Intermediate, and Zero Volume hand gestures recognized for system control.</em>
</p>

<!-- Volume System States -->
<p align="center">
  <img src="Interface Images/Full_Vol_System.png" alt="Full Volume System" width="180"/>
  <img src="Interface Images/Intermediate_Vol_System.png" alt="Intermediate Volume System" width="180"/>
  <img src="Interface Images/Zero_Vol_System.png" alt="Zero Volume System" width="180"/>
  <br/>
  <em>System Volume States: System interface reflecting full, intermediate, and zero volume gestures.</em>
</p>

<!--
E.g.,
![IntelliMotion MediaPipe Demo GIF](link_to_your_gif_or_image.gif)
-->

---

## Key Features

* **Real-time Landmark Detection:** Uses MediaPipe (Pose, Hands, Face Mesh) for fast, accurate detection of body landmarks.
* **Gesture & Pose Recognition:** Recognizes hand gestures and body poses for interactive applications (e.g., volume control, fitness tracking).
* **Motion Analysis:** Tracks movement of specific landmarks, recognizes basic gestures/actions, and provides skeletal visualizations.
* **Versatile Input:** Supports video files (MP4, AVI, etc.) and live webcam streams.
* **Visual Output:** Displays detected landmarks, connections (pose skeleton), and analytics overlays on the video feed.
* **Modular & Extensible:** Easily add new MediaPipe solutions or custom analysis modules.
* **Customizable:** Select MediaPipe solutions, adjust model parameters (confidence thresholds, complexity), and visualization options.

---

## Technologies Used

* **Python 3.8+**
* **OpenCV (cv2):** Video capture, image processing, and display.
* **MediaPipe:** High-fidelity body landmark detection (Pose, Holistic, Hands, Face Mesh).
* **NumPy:** Numerical operations.

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* Pip (Python package installer)
* Git (for cloning the repository)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/TheDeepDelve/IntelliMotion.git
    cd IntelliMotion
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If needed, install manually:
    ```bash
    pip install mediapipe opencv-python numpy
    ```

### Running IntelliMotion

The main script is typically `main.py` (or use the relevant module, e.g., `HandTrackingModule.py`, `PoseTrackingModule.py`).

**Example Usage:**

* **Process a video file with pose detection:**
    ```bash
    python main.py --input PoseVideos/1.mp4 --output output.mp4 --solution pose --show
    ```

* **Process a live webcam feed:**
    ```bash
    python main.py --input 0 --solution pose --show --nosave
    ```

**Command-line Arguments:**

* `--input`: Path to input video file or camera index (e.g., `0` for default webcam).
* `--output` (optional): Path to save processed output video.
* `--solution` (optional): MediaPipe solution to use (`pose`, `hands`, `holistic`).
* `--show` (optional): Display processed video frames in an OpenCV window.
* `--nosave` (optional): Do not save the output video.
* `--model_complexity` (optional): Set model complexity (0, 1, or 2).
* `--min_detection_confidence` (optional): Minimum detection confidence.
* `--min_tracking_confidence` (optional): Minimum tracking confidence.

> **Note:** Update the above arguments to match your `main.py` or module's `argparse` setup.

---

## Configuration

* **MediaPipe Solution Parameters:** Confidence thresholds, model complexity, and other parameters can be set via command-line or in the script.
* **Visualization:** Customize landmark appearance, connections, and overlays in drawing utilities (see `main.py` or `utils/`).
* **Modularity:** Add new gesture/pose modules by extending the codebase (see `HandTrackingModule.py`, `PoseTrackingModule.py`).

---

## Project Structure

```
├── main.py                  # Main entry point (if present)
├── HandTrackingModule.py    # Hand gesture tracking logic
├── PoseTrackingModule.py    # Pose estimation logic
├── Gym_Trainer.py           # Example: fitness/pose application
├── Volume_Control.py        # Example: gesture-based volume control
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── Hand Landmarks.png       # Example output image
├── Pose Landmarks.png       # Example output image
├── Interface Images/        # UI/gesture illustration images
│   ├── Full_Stretched_Arm.png
│   ├── Full_Vol_Gesture.png
│   └── ...
├── PoseVideos/              # Example input videos
│   ├── 1.mp4
│   └── ...
└── ...
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* [MediaPipe](https://mediapipe.dev/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)

---

> **Tip:** For best results, add a GIF or video demo of your system in action! You can use [ScreenToGif](https://www.screentogif.com/) or similar tools to record your screen and upload the GIF to your repository.
