# 🐾 Pest Detector — YOLO-Based Real-Time Pest Detection

A computer vision research project developed for CS5814 at Virginia Tech. This project trains and evaluates multiple YOLO model variants for real-time raccoon and rodent detection on edge devices, and deploys the best-performing model through a React-based monitoring dashboard.

---

## 📁 Project Structure

```
pest-detector/
├── model/      # Dataset processing, training, evaluation, and export notebooks
├── server/     # Python Flask backend (model inference + video stream)
├── ui/         # React/Vite frontend dashboard
└── README.md
```

---

## 🔬 Research Summary

This project addresses the limitations of traditional motion-based pest deterrents by building a focused, single/dual-class object detector targeting **raccoons** and **rodents** (rats and mice). Four YOLO variants were trained and compared across two datasets:

| Model | Dataset | Notes |
|-------|---------|-------|
| YOLOv11n | Raccoon-only | Baseline, 300 epoch |
| YOLOv11n | Raccoon-only / Pest | Extended training, 400 epoch |
| YOLOv8n | Raccoon-only / Pest | **Best overall performer** |
| YOLOv11m | Raccoon-only / Pest | Larger model capacity comparison |

**Key finding:** YOLOv8n achieved the best balance of precision, recall, and robustness across both standard and augmented test conditions, and was selected for edge deployment.

---

## 📦 Datasets

Both datasets were assembled from publicly available Roboflow sources and converted to 3-channel grayscale. All images use YOLO-format bounding box annotations.

- **Raccoon-Only Dataset** — 701 image-label pairs (class 0: raccoon)
- **Pest Dataset** — 1,533 image-label pairs (class 0: raccoon, class 1: rodent)

Sources: [My-Raccoons](https://universe.roboflow.com/proba-7zus4/my-raccoons), [Raccoon](https://public.roboflow.com/object-detection/raccoon), [Rat](https://universe.roboflow.com/tianching/rat-8zffp), [Mouse](https://universe.roboflow.com/mouse-dataset/mouse-v5ogi)

---

## 🧩 Components

### 1. Model (`/model`)

Jupyter notebooks covering the full ML pipeline:

- **Dataset processing** — combining, grayscale conversion, rebalancing (80/10/10 split)
- **Training** — all four YOLO variants via the Ultralytics API, 640×640 input, batch size 16, with augmentations (mosaic, mixup, copy-paste, geometric transforms)
- **Evaluation** — standard test set at 50% confidence / 60% IoU; augmented robustness test using Albumentations (motion blur, Gaussian noise, brightness/contrast, coarse dropout)
- **Synthetic stress test** — 60-second video at 10 FPS simulating fog, rain, lens glare, and overcast conditions
- **Export** — models exported to ONNX, then converted to float16 TFLite via `onnx2tf` on WSL

### 2. Server (`/server`)

Python Flask backend with four modules:

- **Shared Config** — manages global state and hardware mode (desktop vs. Raspberry Pi)
- **Model Loader** — loads the TFLite model, handles inputs and detection outputs
- **Video Feed Handlers** — WSL handler connects to a remote MJPEG feed via network IP; Raspberry Pi handler reads directly from the local hardware driver
- **Server Controller** — Flask app exposing three endpoints:
  - `GET /video_feed` — live detection stream
  - `GET /alerts` — currently detected pests
  - `GET /status` — system performance metrics

### 3. UI (`/ui`)

React + Vite + JavaScript frontend dashboard that polls the three backend endpoints and displays the live video feed, active alerts, and inference performance stats.

---

## ⚙️ Setup

### Requirements

- Python environment with Flask, TFLite runtime, OpenCV
- Node.js (for the UI)
- WSL (Windows Subsystem for Linux) for ONNX → TFLite conversion
- [Motion](https://motion-project.github.io/) for webcam MJPEG streaming on Raspberry Pi
- A webcam or MJPEG video source

### Backend

```bash
# Install Python dependencies
pip install -r server/requirements.txt

# Run the server
python server/server.py
```

### Frontend

```bash
cd ui
npm ci
npm run dev
```

---

## 🖥️ Hardware Tested

| Device | Inference Time | Processing Time | Accuracy |
|--------|---------------|-----------------|----------|
| Desktop (Ryzen 9700x, RTX 4070) | 38.1 ms | 91.6 ms | 99.4% |
| Raspberry Pi 3B+ | 2,809 ms | 3,470 ms | 58.6% |

The Raspberry Pi 3B+ proved insufficient for real-time detection due to its lack of a GPU/NPU and limited RAM (1 GB). A **Raspberry Pi 5** or **NVIDIA Jetson Nano** would be the recommended minimum for a practical deployment.

---

## 📋 Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8n, YOLOv11n, YOLOv11m (Ultralytics) |
| Inference Runtime | TensorFlow Lite (float16) |
| Model Export | ONNX → onnx2tf → TFLite |
| Augmentation | Albumentations |
| Backend | Python, Flask |
| Frontend | React, Vite, JavaScript |
| Video Streaming | Motion (MJPEG) |
| Edge Platform | Raspberry Pi 3B+ / Desktop WSL |

---

## 📄 Citation

Torrin Conrath. 2025. *YOLO-Based Pest Detection for Real-Time Monitoring on Edge Devices.* Virginia Tech, CS5814.
