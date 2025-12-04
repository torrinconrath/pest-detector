import threading
import time
import numpy as np
import cv2
import psutil
import subprocess
from flask import Flask, Response, jsonify
from flask_cors import CORS

from model_loader import YOLODetector
from shared_config import MOTION_URL, MODEL_PATH, GlobalState, ENVIRONMENT

app = Flask(__name__)
CORS(app)

model = None

if ENVIRONMENT == 'raspberry_pi':
    from pi_camera import get_camera_manager
    camera_manager = get_camera_manager()
else:
    from wsl_camera import WSLRemoteCamera
    camera_manager = WSLRemoteCamera()


def get_system_info():
    try:
        cpu_temp = "N/A"
        if ENVIRONMENT == 'raspberry_pi':
            try:
                temp_result = subprocess.run(['vcgencmd', 'measure_temp'],
                                             capture_output=True, text=True)
                if temp_result.returncode == 0:
                    cpu_temp = temp_result.stdout.strip()
            except Exception:
                cpu_temp = "N/A"
        else:
            cpu_temp = "Not available"

        memory = psutil.virtual_memory()
        memory_usage = f"{memory.percent}%"
        cpu_percent = psutil.cpu_percent(interval=0.5)

        return {
            'environment': 'Raspberry Pi' if ENVIRONMENT == 'raspberry_pi' else 'WSL/Desktop',
            'camera_source': 'Local' if ENVIRONMENT == 'raspberry_pi' else 'Remote (Pi)',
            'cpu_temperature': cpu_temp,
            'cpu_usage': f"{cpu_percent}%",
            'memory_usage': memory_usage,
            'timestamp': time.strftime("%H:%M:%S")
        }
    except Exception as e:
        return {'error': str(e)}

def init_model():
    global model
    try:
        model = YOLODetector(MODEL_PATH)
        env_type = 'Raspberry Pi' if ENVIRONMENT == 'raspberry_pi' else 'WSL/Desktop'
        print(f"Model initialized successfully on {env_type}")
        return True
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return False

def inference_loop(stop_event: threading.Event):
    """Inference consumer: waits for newest frame and runs detection on it."""
    global model, camera_manager

    # small history windows
    GlobalState.inference_history = GlobalState.inference_history or []

    while not stop_event.is_set():
        # Wait for a frame (blocks efficiently until camera_manager notifies)
        frame_tuple = camera_manager.wait_for_frame(timeout=1.0)
        if frame_tuple is None:
            # no frame in timeout window; check stop condition and continue
            continue

        # We only need the latest frame - grab it (non-blocking)
        latest = camera_manager.get_latest()
        if latest is None:
            continue

        _, frame = latest

        if model is None:
            # model could have failed to init; skip detection but keep looping
            continue

        # run detection (this can be slow); decoupled from camera thread
        detection_start = time.time()
        try:
            detections, metrics = model.detect(frame, conf_threshold=0.3)
        except Exception as e:
            print(f"Inference error: {e}")
            detections, metrics = [], {}

        detection_time_ms = (time.time() - detection_start) * 1000.0

        # attach timestamps and update global state
        current_time = time.time()
        for d in detections:
            d['timestamp'] = current_time

        GlobalState.latest_detections = detections
        GlobalState.last_detection_time = current_time if detections else GlobalState.last_detection_time

        # metrics: unify shape and store history
        if metrics is None:
            metrics = {}
        metrics['detection_latency_ms'] = detection_time_ms
        # model.detect already returns inference_ms under 'inference_ms' in model_loader
        if 'inference_ms' in metrics:
            GlobalState.inference_history.append(metrics['inference_ms'])
        else:
            GlobalState.inference_history.append(detection_time_ms)

        # cap history length
        if len(GlobalState.inference_history) > 10:
            GlobalState.inference_history.pop(0)

        metrics['avg_inference_ms'] = float(np.mean(GlobalState.inference_history)) if GlobalState.inference_history else 0.0

        # merge in camera metrics if present
        GlobalState.performance_metrics = GlobalState.performance_metrics or {}
        GlobalState.performance_metrics.update(metrics)

        # increment detection count
        if detections:
            GlobalState.detection_count += len(detections)
            env = 'Pi' if ENVIRONMENT == 'raspberry_pi' else 'WSL'
            print(f"[{env}] Found {len(detections)} pests! (Inference: {metrics.get('inference_ms', 0):.1f}ms, Total: {metrics.get('total_ms', 0):.1f}ms)")

    print("Inference loop exiting")

def start_detection_system():
    """Start camera manager and inference thread"""
    print("Starting Pest Detection System...")
    if not init_model():
        print("❌ Cannot start detection system - model failed to load")
        return False

    # start camera capture (if not already)
    camera_manager.start()
    print("📍 Camera manager started")

    # start inference thread
    stop_event = threading.Event()
    inf_thread = threading.Thread(target=inference_loop, args=(stop_event,), daemon=True)
    inf_thread.start()
    print("Inference thread started")

    # return the stop_event and thread so main can manage them
    return {'stop_event': stop_event, 'inference_thread': inf_thread}

@app.route('/video_feed')
def video_feed():
    """Stream latest frames with overlay (non-blocking / uses latest frame)"""
    def gen():
        # new generator-local variables to reduce global lookups
        mgr = camera_manager
        state = GlobalState

        while True:
            latest = mgr.get_latest()
            if latest is None:
                # no frame yet; yield a tiny placeholder or wait briefly
                time.sleep(0.05)
                continue

            _, frame = latest
            # draw detections snapshot (copy frame to avoid modifying shared frame)
            out = frame.copy()

            for det in state.latest_detections:
                try:
                    x1, y1, x2, y2 = det['bbox']
                    label = det.get('label', 'obj')
                    conf = det.get('confidence', 0.0)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(out, text, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception:
                    continue

            # performance overlay
            if state.performance_metrics:
                perf = state.performance_metrics
                fps_text = f"FPS: {perf.get('current_fps', 0):.1f}"
                inf_text = f"Inference: {perf.get('inference_ms', perf.get('detection_latency_ms', 0)):.1f}ms"
                cam_lat_text = f"Camera: {perf.get('camera_latency_ms', 0):.1f}ms"
                platform_text = "Raspberry Pi" if ENVIRONMENT == 'raspberry_pi' else 'WSL/Desktop'

                cv2.putText(out, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(out, inf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(out, cam_lat_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(out, platform_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                # skip if for some reason encoding fails
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    # return current detections (clean up done in caller)
    current = GlobalState.latest_detections or []
    return jsonify({
        "alerts": [
            {
                "id": i,
                "type": det["label"],
                "confidence": round(det["confidence"], 3),
                "time": time.strftime("%H:%M:%S", time.localtime(det.get('timestamp', time.time())))
            }
            for i, det in enumerate(current)
        ],
        "total_detections": GlobalState.detection_count,
        "active_alerts": len(current)
    })

@app.route('/status')
def get_status():
    # system info
    sys_info = get_system_info()
    last_detection_str = "Never"
    if GlobalState.last_detection_time > 0:
        last_detection_str = time.strftime("%H:%M:%S", time.localtime(GlobalState.last_detection_time))

    status = {
        "camera": "connected" if GlobalState.camera_connected else "disconnected",
        "model": "loaded" if model else "failed",
        "active_alerts": len(GlobalState.latest_detections),
        "total_detections": GlobalState.detection_count,
        "last_detection": last_detection_str,
        "frame_counter": GlobalState.frame_counter,
        "performance": {
            **(GlobalState.performance_metrics or {}),
            "fps_history": GlobalState.fps_history,
            "inference_history": GlobalState.inference_history
        },
        "system": sys_info,
        "timestamp": time.strftime("%H:%M:%S")
    }
    return jsonify(status)

@app.route('/')
def index():
    return jsonify({
        "message": "Pest Detection Server",
        "environment": "Raspberry Pi" if ENVIRONMENT == 'raspberry_pi' else 'WSL/Desktop',
        "camera_source": "Local" if ENVIRONMENT == 'raspberry_pi' else 'Remote (Pi)',
        "endpoints": {
            "/video_feed": "Live video with detections and performance overlay",
            "/alerts": "Current alerts (JSON)",
            "/status": "System status with FPS, latency, and inference time"
        }
    })

if __name__ == "__main__":
    # start everything
    result = start_detection_system()
    if not result:
        print("Failed to start system")
        exit(1)

    stop_event = result['stop_event']
    try:
        env_type = 'Raspberry Pi' if ENVIRONMENT == 'raspberry_pi' else 'WSL/Desktop'
        print(f"Starting server on {env_type}...")
        print("Endpoints:")
        print("/video_feed - Live video with detections")
        print("/alerts - Current alerts")
        print("/status - Performance metrics")
        app.run(host='0.0.0.0', port=8082, debug=False, threaded=True)
    finally:
        # signal inference thread to stop and stop camera
        stop_event.set()
        camera_manager.stop()
        print("Server shutdown complete")
