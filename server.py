# server.py
from flask import Flask, Response, stream_with_context
from model_loader import YOLODetector
import cv2
import json
import time

app = Flask(__name__)
model = YOLODetector("yolov11n.onnx")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 10)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            success, frame = camera.read()
            if not success:
                continue

            detections = model.detect(frame)

            # Draw boxes
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.1)  # limit to ~10 FPS
    return Response(stream_with_context(generate()), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/alerts")
def alerts():
    def event_stream():
        while True:
            success, frame = camera.read()
            if not success:
                continue
            detections = model.detect(frame)
            for det in detections:
                yield f"data: {json.dumps({'type': det['label'], 'confidence': det['confidence']})}\n\n"
            time.sleep(0.2)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True, threaded=True)
