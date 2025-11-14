import cv2
import threading
import time
from flask import Flask, Response, stream_with_context
from model_loader import YOLODetector
import json

app = Flask(__name__)
model = YOLODetector("yolov11n.onnx")

# Shared variables
frame_lock = threading.Lock()
latest_frame = None
latest_detections = []

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS

def capture_frames():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.01)  # small sleep to avoid 100% CPU

def detect_objects():
    global latest_detections
    while True:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        with frame_lock:
            frame_copy = latest_frame.copy()
        detections = model.detect(frame_copy)
        latest_detections = detections
        time.sleep(0.2)  # run detection every 0.2s (~5 FPS)

# Start background threads
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=detect_objects, daemon=True).start()

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if latest_frame is None:
                continue
            with frame_lock:
                frame_copy = latest_frame.copy()
                detections_copy = latest_detections.copy()

            # Draw boxes
            for det in detections_copy:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['confidence']:.2f}"
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_copy, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame_copy)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.03)  # ~30 FPS
    return Response(stream_with_context(generate()), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/alerts")
def alerts():
    def event_stream():
        while True:
            for det in latest_detections:
                yield f"data: {json.dumps({'type': det['label'], 'confidence': det['confidence']})}\n\n"
            time.sleep(0.2)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)
