# raspberry-pi5-edgetpu-inference

## 🧠 Project Overview

We present a unified benchmarking and deployment framework for running multiple quantized vision models on Raspberry Pi 5 using Google's Edge TPU USB accelerator. This system integrates three key computer vision tasks:

- 🐦 **Bird Species Classification** (MobileNetV2, 224×224)
- 🙂 **Face Detection** (BlazeFace, 320×320)
- 😐 **Facial Expression Recognition (FER)**  
  - Model 1: 64×64 grayscale (trained on the FER+ dataset)  
  - Model 2: 112×112 RGB (based on the MobileFaceNet architecture)

