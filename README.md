# Conducting_Tutor

# Project Overview

This project processes video data to detect conducting movements and generate visualizations. Below is a brief description of each file in the project:

- **beat_filter.py**: Contains functions for filtering significant points (beats) from the detected peaks and valleys in the movement data. It analyzes the frame data to identify and return filtered significant beats.

- **graphs.py**: Responsible for generating various analysis graphs from the processed data, including plots for beat detection, hand movements, and swaying detection.

- **imports.py**: Centralized file for importing necessary libraries and modules, including standard libraries, Mediapipe, and local modules.

- **main.py**: The main execution point of the program, handling the first and second passes through the video for detecting movements and generating visualizations.

- **mirror.py**: Implements the `mirrorDetection` class, which tracks the coordinates of the left and right hands for analyzing mirror movements.

- **mp_declaration.py**: Contains the `mediaPipeDeclaration` class, which sets up and manages the Mediapipe pose landmark detection.

- **names.py**: Provides utility functions for generating names for video files, output files, and plots based on the input video file.

- **p_stage1.py**: Handles the first stage of video processing, including frame processing and landmark detection, as well as user input for controlling the processing.

- **pattern.py**: Implements the `patternDetection` class, which analyzes the detected beats to identify movement patterns based on the coordinates of the beats.

- **p_stage2.py**: Manages the second stage of video processing, displaying the annotated frames with detected beats and generating BPM data.

- **sway.py**: Implements the `swayingDetection` class, which detects swaying movements based on the calculated midpoints of the hands.
