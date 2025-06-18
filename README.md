# FocuSense - Real-Time Engagement Detection

## Overview
FocuSense is an advanced facial recognition system developed to analyze user engagement in real-time during video calls. Utilizing OpenCV and dlib libraries, it accurately processes video frames to detect facial landmarks and calculate the Eye Aspect Ratio (EAR) for engagement assessment.

## Installation
1. **Clone the Repository**: `git clone https://github.com/dhruvvaz/FocuSense.git`
2. **Install Dependencies**: You will need OpenCV, dlib, and scipy. Use `pip install opencv-python dlib scipy`.

## Setup
- Download the required model file:
  - [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
- Place this file in the same directory as the Python scripts or update the path in the code.

## Usage
Run the script using Python. The program captures video from the webcam and displays the engagement level in real-time.

## Contributing
Contributions to FocuSense are welcome. Please read the CONTRIBUTING.md for guidelines.
