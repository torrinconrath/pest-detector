import numpy as np
import cv2
import time
from tflite_runtime.interpreter import Interpreter

CLASS_NAMES = ["raccoon", "rodent"]

class YOLODetector:
    def __init__(self, model_path: str):
        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"✅ TFLite Model loaded: {model_path}")
            print(f"Input: {self.input_details[0]['shape']}")
            print(f"Output: {self.output_details[0]['shape']}")
            print(f"Expected classes: {CLASS_NAMES}")

        except Exception as e:
            print(f"❌ Failed to load TFLite model: {e}")
            raise

    def preprocess(self, frame):
        """Preprocess frame for model"""
        input_shape = self.input_details[0]['shape']
        ih, iw = input_shape[1], input_shape[2]
        
        img = cv2.resize(frame, (iw, ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    def detect(self, frame, conf_threshold=0.5):
        """Run detection and return metrics"""
        start_time = time.time()
        
        try:
            preprocess_start = time.time()
            input_img = self.preprocess(frame)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            inference_start = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
            self.interpreter.invoke()
            inference_time = (time.time() - inference_start) * 1000
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            output = np.squeeze(output)
            
            process_start = time.time()
            detections = self.process_predictions(output, frame.shape, conf_threshold)
            process_time = (time.time() - process_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            metrics = {
                'preprocess_ms': preprocess_time,
                'inference_ms': inference_time,
                'process_ms': process_time,
                'total_ms': total_time,
                'fps': 1000 / total_time if total_time > 0 else 0
            }
            
            return detections, metrics

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return [], {}

    def process_predictions(self, predictions, frame_shape, conf_threshold):
        """Process model predictions with selective sigmoid"""
        detections = []
        h, w = frame_shape[:2]
        
        if predictions.shape[0] == 6:
            predictions = predictions.T
        
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            if len(pred) >= 6:
                # Fetch parameters
                x_center, y_center, width, height = pred[0:4]
                class_scores = pred[4:4 + len(CLASS_NAMES)]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if (confidence > conf_threshold and 
                    x_center >= 0 and x_center <= 640 and
                    y_center >= 0 and y_center <= 640 and
                    width > 10 and height > 10):
                    
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    if bbox_width > 20 and bbox_height > 20:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(confidence),
                            "class_id": int(class_id),
                            "label": CLASS_NAMES[class_id]
                        })
        
        return self.non_max_suppression(detections)

    def non_max_suppression(self, detections, iou_threshold=0.6):
        """Remove overlapping detections"""
        if len(detections) == 0:
            return []

        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                det for det in detections
                if not (det['class_id'] == best['class_id'] and 
                       self.iou(det['bbox'], best['bbox']) > iou_threshold)
            ]
        
        return keep

    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
