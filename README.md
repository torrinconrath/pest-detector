# 🐾 Pest Detector — YOLO-Based Real-Time Pest Detection

A real-time computer vision system for detecting raccoons and rodents using lightweight YOLO models, deployed through a React-based monitoring dashboard. Built to address the limitations of traditional motion sensors, which trigger on anything that moves, by using a focused object detector that only alerts on actual pests.

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

## 🧩 Components

### 1. Model (`/model`)

Jupyter notebooks covering the full ML pipeline:

- **Dataset processing** — combining sources, grayscale conversion, rebalancing (80/10/10 split)
- **Training** — YOLO variants via the Ultralytics API, 640×640 input, batch size 16, with augmentations (mosaic, mixup, copy-paste, geometric transforms)
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

The Raspberry Pi 3B+ proved insufficient for real-time detection due to its lack of a GPU/NPU and limited RAM. A **Raspberry Pi 5** or **NVIDIA Jetson Nano** would be the recommended minimum for a practical edge deployment.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8n, YOLOv11n, YOLOv11m (Ultralytics) |
| Inference Runtime | TensorFlow Lite (float16) |
| Model Export | ONNX → onnx2tf → TFLite |
| Augmentation | Albumentations |
| Backend | Python, Flask |
| Frontend | React, Vite, JavaScript |
| Video Streaming | Motion (MJPEG) |
| Edge Platform | Raspberry Pi 3B+ / Desktop (WSL) |
