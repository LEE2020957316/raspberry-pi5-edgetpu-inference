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
├── models/            # Precompiled .tflite and .edgetpu.tflite models
├── scripts/           # Python scripts for real-time inference and static tests
├── images/            # Sample test images 
└── README.md          # This file

## 🚀 Quick Start
1. **Set up environments**

   - Python **3.11**: used for camera capture and display (main system Python on Raspberry Pi 5)
   - Python **3.9 (virtual environment)**: used for Edge TPU inference (required by PyCoral runtime)

   > ⚠️ PyCoral only provides prebuilt wheels for Python 3.9. Make sure to create a dedicated virtual environment for inference.

📊 Performance Summary

