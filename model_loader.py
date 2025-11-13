# model_loader.py
import onnxruntime as ort
import numpy as np
import cv2
import time

# Define class names consistent with your build_model.ipynb
CLASS_NAMES = ["raccoon", "rat", "mouse"]

class YOLODetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, frame, conf_threshold=0.4):
        img = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: img})[0]

        detections = []
        for det in outputs:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            if conf < conf_threshold:
                continue
            cls_id = int(cls_id)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Object {cls_id}"
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class_id": cls_id,
                "label": label,
            })
        return detections
