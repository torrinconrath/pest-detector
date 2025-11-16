# model_loader.py
import onnxruntime as ort
import numpy as np
import cv2
import time

# Define class names for 2-class system (raccoon, rodent)
CLASS_NAMES = ["raccoon", "rodent"]

class YOLODetector:
    def __init__(self, model_path: str):
        # Use CPU provider for Raspberry Pi compatibility
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"✅ Model loaded: {model_path}")
        print(f"✅ Input name: {self.input_name}")
        print(f"✅ Output name: {self.output_name}")

    def preprocess(self, frame):
        # Convert to grayscale for consistency with training
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to 3-channel (but still grayscale)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Resize to model input size
        img = cv2.resize(gray_3ch, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, frame, conf_threshold=0.3):
        try:
            img = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: img})[0]

            detections = []
            # Handle different output formats from YOLO models
            if outputs.ndim == 3:  # Newer YOLO format (1, 84, 8400)
                outputs = outputs[0]  # Remove batch dimension
                # Convert from center format to corner format if needed
                for i in range(outputs.shape[1]):
                    det = outputs[:, i]
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        if conf >= conf_threshold:
                            cls_id = int(cls_id)
                            if cls_id < len(CLASS_NAMES):
                                label = CLASS_NAMES[cls_id]
                                detections.append({
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(conf),
                                    "class_id": cls_id,
                                    "label": label,
                                })
            else:  # Older format or custom export
                for det in outputs:
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        if conf >= conf_threshold:
                            cls_id = int(cls_id)
                            if cls_id < len(CLASS_NAMES):
                                label = CLASS_NAMES[cls_id]
                                detections.append({
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(conf),
                                    "class_id": cls_id,
                                    "label": label,
                                })
            return detections
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []