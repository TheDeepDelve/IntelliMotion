# IntelliMotion: Real-Time Hand Gesture and Pose Tracking Using OpenCV and MediaPipe

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Made with OpenCV](https://img.shields.io/badge/Made%20with-OpenCV-brightgreen.svg)](https://opencv.org/)
[![Powered by MediaPipe](https://img.shields.io/badge/Powered%20by-MediaPipe-orange.svg)](https://mediapipe.dev/)

IntelliMotion is a Python-based application for real-time intelligent motion analysis using Google's MediaPipe and OpenCV. It detects and tracks human body landmarks (pose and hands) for gesture recognition, pose estimation, and interactive applications. The modular design allows for easy extension and customization.

---

## Demo

<!-- Replace the below with your own GIF or video demo if available -->
<p align="center">
  <img src="Hand Landmarks.png" alt="Hand Landmarks Example" width="350" height="350"/>
  <br/>
  <em>Hand Landmarks Example: MediaPipe detects and annotates key points on the hand in real time.</em>
</p>

<p align="center">
  <img src="Pose Landmarks.png" alt="Pose Landmarks Example" width="350" height="350"/>
  <br/>
  <em>Pose Landmarks Example: Full-body pose estimation with skeletal connections visualized.</em>
</p>

---
## Table of Contents

* [Demo](#demo)
  * [Gesture & System States Gallery](#gesture--system-states-gallery)
* [Key Features](#key-features)
* [Technologies Used](#technologies-used)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Running IntelliMotion](#running-intellimotion)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Project Flowcharts](#project-flowcharts)
* [Contributing](#contributing)
* [Contact](#contact)

---

### Gesture & System States Gallery

<!-- Stretched Arm States -->
<p align="center">
  <img src="Interface Images/Full_Stretched_Arm.png" alt="Full Stretched Arm" width="220" height="280"/>
  <img src="Interface Images/Partial_Stretched_Arm.png" alt="Partial Stretched Arm" width="220" height="280"/>
  <br/>
  <em>Left: Full Stretched Arm - Detected pose with arm fully extended.<br>Right: Partial Stretched Arm - Detected pose with arm partially extended.</em>
</p>

<!-- Volume Gestures -->
<p align="center">
  <img src="Interface Images/Full_Vol_Gesture.png" alt="Full Volume Gesture" width="180" height="200"/>
  <img src="Interface Images/Intermediate_Vol_Gesture.png" alt="Intermediate Volume Gesture" width="180" height="200"/>
  <img src="Interface Images/Zero_Vol_Gesture.png" alt="Zero Volume Gesture" width="180" height="200"/>
  <br/>
  <em>Volume Gestures: Full, Intermediate, and Zero Volume hand gestures recognized for system control.</em>
</p>

<!-- Volume System States -->
<p align="center">
  <img src="Interface Images/Full_Vol_System.png" alt="Full Volume System" width="180" height="120"/>
  <img src="Interface Images/Intermediate_Vol_System.png" alt="Intermediate Volume System" width="180" height="120"/>
  <img src="Interface Images/Zero_Vol_System.png" alt="Zero Volume System" width="180" height="120"/>
  <br/>
  <em>System Volume States: System interface reflecting full, intermediate, and zero volume gestures.</em>
</p>

<!--
E.g.,
![IntelliMotion MediaPipe Demo GIF](link_to_your_gif_or_image.gif)
-->

---

## Key Features

* **Real-time Landmark Detection:** Uses MediaPipe (Pose, Hands) for fast, accurate detection of body landmarks.
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
* **MediaPipe:** High-fidelity body landmark detection (Pose, Holistic, Hands).
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

IntelliMotion is organized into separate modules for different use cases. There are no command-line arguments required. Simply run the relevant Python script for your desired functionality:

- **To control system volume using hand gestures:**
    ```bash
    python Volume_Control.py
    ```
    This will launch the volume control interface, allowing you to adjust your system's volume with hand gestures in real time.

- **To track your pose for gym/fitness applications:**
    ```bash
    python Gym_Trainer.py
    ```
    This will start the gym trainer module, providing real-time pose tracking and feedback for your workouts.

You can also explore or extend other modules (such as `HandTrackingModule.py` or `PoseTrackingModule.py`) for additional or custom functionality.

---

## Configuration

* **MediaPipe Solution Parameters:** Confidence thresholds, model complexity, and other parameters can be set via command-line or in the script.
* **Visualization:** Customize landmark appearance, connections, and overlays in drawing utilities (see `main.py` or `utils/`).
* **Modularity:** Add new gesture/pose modules by extending the codebase (see `HandTrackingModule.py`, `PoseTrackingModule.py`).

---

## Project Structure

```
├── main.py                  
├── HandTrackingModule.py    # Hand gesture tracking logic
├── PoseTrackingModule.py    # Pose estimation logic
├── Gym_Trainer.py           # Pose application
├── Volume_Control.py        # Gesture-based volume control application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── Hand Landmarks.png       # MediaPipe Hand Lanmarks
├── Pose Landmarks.png       # MediaPipe Pose Landmarks
├── Interface Images/        # UI/gesture illustration images
│   ├── Full_Stretched_Arm.png
│   ├── Full_Vol_Gesture.png
│   └── ...
├── PoseVideos/              # Sample input videos
│   ├── 1.mp4
│   └── ...
└── ...
```


---

## Project Flowcharts

To better understand the workflow and core processes of IntelliMotion, refer to the following flowcharts:

<p align="center">
  <img src="Flowcharts/Work Flow.png" alt="Overall Workflow" width="500"/>
  <br/>
  <em>Overall Workflow: High-level overview of the IntelliMotion system pipeline.</em>
</p>

<p align="center">
  <img src="Flowcharts/Data PreProcessing.jpg" alt="Data Preprocessing Flowchart" width="400"/>
  <br/>
  <em>Data Preprocessing: Steps involved in preparing input data for analysis.</em>
</p>

<p align="center">
  <img src="Flowcharts/Feature Extraction.jpg" alt="Feature Extraction Flowchart" width="400"/>
  <br/>
  <em>Feature Extraction: How key features are extracted from video frames or images.</em>
</p>

<p align="center">
  <img src="Flowcharts/Gesture Classification.jpg" alt="Gesture Classification Flowchart" width="400"/>
  <br/>
  <em>Gesture Classification: The process of classifying gestures based on extracted features.</em>
</p>

---

## Contributing

Contributions, issues, and feature requests are welcome!

To contribute:
1. **Fork the Project**
2. **Create your Feature Branch**
    ```bash
    git checkout -b feature/AmazingFeature
    ```
3. **Commit your Changes**
    ```bash
    git commit -m 'Add some AmazingFeature'
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature/AmazingFeature
    ```
5. **Open a Pull Request**

Feel free to open issues for bugs, suggestions, or questions. We appreciate your help in making IntelliMotion better!

---

## Contact

Harsh Deep - [LinkedIn](https://www.linkedin.com/in/harshdeep7199/) - [Email](harshdeep7199@gmail.com)


<!-- > **Tip:** For best results, add a GIF or video demo of your system in action! You can use [ScreenToGif](https://www.screentogif.com/) or similar tools to record your screen and upload the GIF to your repository. -->
