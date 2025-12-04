import os
import platform

def detect_environment():
    """Detect if we're running on Raspberry Pi or other system"""
    # Check for Raspberry Pi
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                return 'raspberry_pi'
    
    return 'other'

# Environment detection
ENVIRONMENT = detect_environment()

# Raspberry Pi Configuration (ALWAYS uses local camera)
if ENVIRONMENT == 'raspberry_pi':
    MOTION_URL = "http://localhost:8081"
    MODEL_PATH = os.path.expanduser("~/pest_detection/models/pest_model_float16.tflite") # Where I put my model
    print("Environment: Raspberry Pi (Local camera)")
else:
    # WSL/Desktop - connects to Raspberry Pi's camera stream
    RASPBERRY_PI_IP = "10.0.0.113"  # Pi's IP
    MOTION_URL = f"http://{RASPBERRY_PI_IP}:8081"
    MODEL_PATH = os.path.expanduser("~/pest_models/saved_model/pest_model_float16.tflite") # Where I put my model
    print(f"Environment: WSL/Desktop (Remote camera from Pi: {RASPBERRY_PI_IP})")

# Global state
class GlobalState:
    latest_detections = []
    detection_count = 0
    camera_connected = False
    performance_metrics = {}
    frame_counter = 0
    last_detection_time = 0
    fps_history = []
    inference_history = []
