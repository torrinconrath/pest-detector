import cv2
import threading
import time
import json
import os
from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import numpy as np  # Add CORS support
from model_loader import YOLODetector

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Try different camera indices for flexibility
def init_camera():
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        try:
            camera = cv2.VideoCapture(camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            camera.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for RPi 3B+
            
            # Test if camera works
            ret, frame = camera.read()
            if ret and frame is not None:
                print(f"✅ Camera found at index {camera_index}")
                return camera
            camera.release()
        except Exception as e:
            print(f"❌ Camera index {camera_index} failed: {e}")
    
    print("❌ No camera found! Using fallback mode.")
    return None

# Initialize camera
camera = init_camera()

# Load model
try:
    model = YOLODetector("yolov11n.onnx")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# Shared variables
frame_lock = threading.Lock()
latest_frame = None
latest_detections = []
frame_count = 0

def create_test_frame():
    """Create a test frame when no camera is available"""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "NO CAMERA - TEST MODE", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def capture_frames():
    global latest_frame, frame_count
    while True:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                with frame_lock:
                    latest_frame = frame.copy()
                    frame_count += 1
            else:
                print("❌ Camera read failed")
                with frame_lock:
                    latest_frame = create_test_frame()
        else:
            # Fallback: create test frames
            with frame_lock:
                latest_frame = create_test_frame()
            frame_count += 1
        
        time.sleep(0.1)  # 10 FPS max for RPi

def detect_objects():
    global latest_detections
    detection_count = 0
    
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        with frame_lock:
            frame_copy = latest_frame.copy()
            current_frame_count = frame_count
        
        # Only process every 5th frame for performance (~2 FPS detection)
        if detection_count % 5 == 0 and model is not None:
            try:
                detections = model.detect(frame_copy, conf_threshold=0.3)
                latest_detections = detections
                
                # Print detection info occasionally
                if detections and detection_count % 20 == 0:
                    print(f"📹 Frame {current_frame_count}: Detected {len(detections)} objects")
                    for det in detections:
                        print(f"   - {det['label']}: {det['confidence']:.2f}")
            except Exception as e:
                print(f"❌ Detection error: {e}")
                latest_detections = []
        
        detection_count += 1
        time.sleep(0.1)  # 10 FPS detection loop

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if latest_frame is None:
                time.sleep(0.1)
                continue
                
            with frame_lock:
                frame_copy = latest_frame.copy()
                detections_copy = latest_detections.copy()

            # Draw detection boxes
            for det in detections_copy:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['confidence']:.2f}"
                
                # Color code by class
                color = (0, 255, 0) if det["class_id"] == 0 else (255, 0, 0)  # Green for raccoon, Blue for rodent
                
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode frame as JPEG
            success, buffer = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                continue
                
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.1)  # ~10 FPS stream

    return Response(stream_with_context(generate()), 
                   mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/alerts")
def alerts():
    def event_stream():
        last_sent = {}
        alert_id = 0
        
        while True:
            current_time = time.time()
            for det in latest_detections:
                # Only send alerts for high confidence detections
                if det["confidence"] > 0.5:
                    alert_key = f"{det['label']}_{det['bbox'][0]}_{det['bbox'][1]}"
                    
                    # Throttle alerts (max 1 per 10 seconds per object)
                    if (alert_key not in last_sent or 
                        current_time - last_sent[alert_key] > 10):
                        
                        alert_data = {
                            "type": det["label"],
                            "confidence": det["confidence"],
                            "time": time.strftime("%H:%M:%S")
                        }
                        
                        yield f"data: {json.dumps(alert_data)}\n\n"
                        last_sent[alert_key] = current_time
                        alert_id += 1
            
            # Send a heartbeat to keep connection alive
            yield "data: {\"type\": \"heartbeat\"}\n\n"
            time.sleep(1)  # Check every second

    return Response(stream_with_context(event_stream()), 
                   mimetype="text/event-stream")

@app.route("/status")
def status():
    camera_status = "connected" if camera and camera.isOpened() else "disconnected"
    model_status = "loaded" if model else "failed"
    
    return {
        "camera": camera_status,
        "model": model_status,
        "detections": len(latest_detections),
        "frame_count": frame_count,
        "timestamp": time.time()
    }

# Simple HTML page for testing
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pest Detection Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>Pest Detection Server</h1>
        <p>This server provides video feed and alerts for the React dashboard.</p>
        
        <div class="status" id="status">Loading status...</div>
        
        <h3>Endpoints:</h3>
        <ul>
            <li><a href="/video_feed" target="_blank">Video Feed</a></li>
            <li><a href="/alerts" target="_blank">Alerts Stream</a></li>
            <li><a href="/status" target="_blank">Status API</a></li>
        </ul>
        
        <script>
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerHTML = 
                        `Camera: <strong>${data.camera}</strong> | ` +
                        `Model: <strong>${data.model}</strong> | ` +
                        `Detections: <strong>${data.detections}</strong>`;
                });
        </script>
    </body>
    </html>
    """

# Start background threads
print("🚀 Starting background threads...")
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=detect_objects, daemon=True).start()
print("✅ Background threads started")

if __name__ == "__main__":
    print("🌐 Starting web server on http://0.0.0.0:8081")
    print("📹 Video feed: http://localhost:8081/video_feed")
    print("🚨 Alerts: http://localhost:8081/alerts")
    print("📊 Status: http://localhost:8081/status")
    app.run(host="0.0.0.0", port=8081, debug=False, threaded=True)