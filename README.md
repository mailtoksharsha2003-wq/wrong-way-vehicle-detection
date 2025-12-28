# Wrong-Way Vehicle Detection from Moving Dashcam Videos

This project implements an advanced **computer vision pipeline** to automatically detect **wrong-way vehicle movement** from **moving vehicle dashcam footage** using deep learning, multi-object tracking, and ego-motion compensation.

The system is designed to handle real-world challenges such as camera motion, motion blur, small objects, and noisy detections commonly present in dashcam videos.

---

## Problem Statement

Wrong-side driving is a major cause of road accidents and traffic violations.  
Detecting such violations from **moving dashcam videos** is challenging due to:

- Continuous camera motion (ego-motion)
- Motion blur
- Dynamic background
- Varying vehicle speeds and sizes

Manual inspection of long dashcam videos is inefficient and error-prone.

---

## System Overview

Input Dashcam Video
↓
Vehicle Detection (YOLOv8)
↓
Multi-Object Tracking (SORT)
↓
Centroid & Velocity History
↓
Ego-Motion Compensation
↓
Direction Classification
↓
Temporal Confirmation (Hysteresis)
↓
Wrong-Way Violation Detection
↓
Evidence Saving (Vehicle Crops + Annotated Video)

---

## Key Features

- YOLOv8-based vehicle detection (car, motorcycle, bus, truck)
- SORT-based multi-object tracking
- Designed specifically for **moving dashcam footage**
- Ego-motion compensation using:
  - ORB feature matching with RANSAC
  - Optical flow fallback (Lucas–Kanade)
- Multi-scale detection for small or distant vehicles
- Motion blur detection and quality filtering
- Temporal hysteresis for stable direction confirmation
- Automatic saving of confirmed wrong-way vehicle crops
- Annotated output video with direction visualization

---

## Detection Logic

- Vehicles are tracked across consecutive frames
- Direction is estimated using smoothed centroid motion over time
- Camera-induced motion is removed using ego-motion estimation
- A vehicle is classified as **Wrong-Way** only after:
  - Consistent direction over multiple frames
  - Passing blur and quality checks
  - Temporal confirmation thresholds

This significantly reduces false positives in real-world traffic scenarios.

---

## Usage

```bash
python wsd.py
Configuration

Input and output paths can be modified inside the script:

video_path = "data/input_video.mp4"
output_root = "outputs"

Outputs

Annotated output video showing tracked vehicles and directions

Cropped images of confirmed wrong-way vehicles

Console summary statistics after execution
