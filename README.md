# raspberry-pi5-edgetpu-inference

## 🧠 Project Overview

We present a unified benchmarking and deployment framework for running multiple quantized vision models on Raspberry Pi 5 using Google's Edge TPU USB accelerator. This system integrates three key computer vision tasks:

- 🐦 **Bird Species Classification** (MobileNetV2, 224×224)
- 🙂 **Face Detection** (BlazeFace, 320×320)
- 😐 **Facial Expression Recognition (FER)**  
  - Model 1: 64×64 grayscale (trained on the FER+ dataset)  
  - Model 2: 112×112 RGB (based on the MobileFaceNet architecture)


## 🔧 Features

- Dual-Python architecture using ZeroMQ (Python 3.11 for camera + Python 3.9 for TPU inference)
- Real-time camera processing at up to 10.9 FPS (FD+FER, 112×112)
- Edge TPU acceleration using PyCoral and precompiled `.edgetpu.tflite` models
- Supports multiple faces per frame, with scalable FER throughput

## 🖥️ Inference Pipeline Architecture

[Raspberry Pi Camera] → Python 3.11 → JPEG → ZeroMQ → Python 3.9 → [Edge TPU] → FD + FER → Result → ZeroMQ → Python 3.11 → Display

This dual-process pipeline resolves version incompatibility between the Picamera2 library (Python 3.11) and the PyCoral runtime (Python 3.9), while ensuring low-latency communication and real-time feedback.

## 📂 Repository Structure

```
├── models/              # Precompiled .tflite and .edgetpu.tflite models  
├── scripts/             # Python scripts for real-time inference and static tests  
├── images/              # Sample test images  
└── README.md            # This file
```
## 🚀 Quick Start
1. **Set up environments**

   - Python **3.11**: used for camera capture and display (main system Python on Raspberry Pi 5)
   - Python **3.9 (virtual environment)**: used for Edge TPU inference (required by PyCoral runtime)

   > ⚠️ PyCoral only provides prebuilt wheels for Python 3.9. Make sure to create a dedicated virtual environment for inference.

## 📊 Model Performance Summary

| Model         | CPU FPS | TPU FPS |
| ------------- | ------- | ------- |
| MobileNetV2   | 56.7    | 225.9   |
| Face Detector | 29.3    | 122.5   |
| FER (112×112) | 36.2    | 201.4   |
| FER (64×64)   | 53.4    | 154.6   |

## 📊 Real-Time Inference Performance (Single Face Input)

Average **back-end multitask inference throughput**, measured on Raspberry Pi 5:

- **FD + FER (112×112)**: 10.9 FPS  
- **FD + FER (64×64)**: 8.7 FPS  

> ⚙️ These values represent the average processing speed of the **back-end multitask pipeline** — from the moment when the Python 3.9 virtual environment receives a JPEG image frame (via ZeroMQ), to the completion of face detection and expression recognition, and returning the result to Python 3.11.  
> 
> ⚠️ This definition **excludes** front-end overhead such as camera capture (Picamera2), JPEG encoding, and final display. It reflects only the throughput of the actual inference pipeline.

## 📄 Citation

This repository contains the implementation described in a paper currently under submission to **FPT 2025**.

Please do not cite until the paper is formally accepted.
