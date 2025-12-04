"""
Synthetic Test: Realistic Pest Detection Simulation
----------------------------------------------------------
Focused synthetic environment to test YOLOv11n in two modes:
1. RACCOON_ONLY: Detects only raccoons (class 0)
2. GENERAL_PEST: Detects both raccoons (0) and rodents (1)

Simulates realistic conditions (fog, rain, glare, overcast) and applies 
grayscale-only image processing to match model training conditions.
"""

import os
import cv2
import glob
import random
import numpy as np
from ultralytics import YOLO
from pathlib import Path

FRAME_SIZE = (640, 640)
FPS = 10
DURATION = 60

# === MODE CONFIGURATION ===
class TestMode:
    RACCOON_ONLY = "raccoon_only"
    GENERAL_PEST = "general_pest"

# Model paths for each mode
MODEL_PATHS = {
    TestMode.RACCOON_ONLY: "runs/detect/train7/weights/best.pt",
    TestMode.GENERAL_PEST: "runs/detect/train8/weights/best.pt"
}

# Test image directories for each mode
TEST_IMAGES_DIRS = {
    TestMode.RACCOON_ONLY: "rebalanced_raccoon_dataset/test/images",
    TestMode.GENERAL_PEST: "rebalanced_dataset/test/images"
}

# === ENVIRONMENT PROFILES ===
class EnvironmentProfile:
    MODES = ["clear", "foggy", "rainy", "lens_glare", "overcast"]
    MODE_WEIGHTS = [0.5, 0.2, 0.2, 0.10, 0.10]

    CONFIDENCE_THRESHOLDS = {
        "clear": 0.3,
        "foggy": 0.35,
        "rainy": 0.4,
        "lens_glare": 0.35,
        "overcast": 0.3
    }

    PEST_PROB = 0.25  # Chance of spawning a pest in a frame
    IOU_THRESHOLD = 0.2 

config = EnvironmentProfile()

# === ENVIRONMENTAL SIMULATION ===
class GrayscaleEnvironment:
    def apply_effect(self, img, mode):
        """Apply environment effects to grayscale image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        if mode == "clear":
            gray *= 0.45
            gray += np.random.normal(0, 6, gray.shape)

        elif mode == "foggy":
            gray *= 0.5
            gray = cv2.GaussianBlur(gray, (9, 9), 3)
            gray += np.random.normal(0, 3, gray.shape)

        elif mode == "rainy":
            gray *= 0.6
            # Add rain streaks
            for _ in range(40):
                x = random.randint(0, gray.shape[1] - 1)
                length = random.randint(10, 25)
                intensity = random.randint(10, 20)
                for j in range(length):
                    if x + j < gray.shape[1]:
                        pos = min(gray.shape[0]-1, random.randint(0, gray.shape[0]-1) + j)
                        gray[pos, x+j] += intensity

            gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

        elif mode == "lens_glare":
            gray *= 0.7
            overlay = np.zeros_like(gray)
            # Add glare circles
            for _ in range(random.randint(1, 2)):
                center = (random.randint(0, gray.shape[1]), random.randint(0, gray.shape[0]))
                radius = random.randint(50, 120)
                cv2.circle(overlay, center, radius, random.randint(180, 255), -1)
                cv2.GaussianBlur(overlay, (0, 0), 30, overlay)
            gray = cv2.addWeighted(gray, 1.0, overlay, 0.3, 0)

        elif mode == "overcast":
            gray *= 0.8
            gray = cv2.equalizeHist(gray.astype(np.uint8)).astype(np.float32)

        # Add slight Gaussian noise
        gray += np.random.normal(0, 2, gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# === PEST OVERLAY ===
class PestOverlay:
    def __init__(self, frame_size, test_images_dir, mode):
        self.frame_size = frame_size
        self.mode = mode
        self.images = self.load_images(test_images_dir)
        self.class_names = self.get_class_names()

    def get_class_names(self):
        """Get class names based on mode"""
        if self.mode == TestMode.RACCOON_ONLY:
            return {0: "raccoon"}
        else:
            return {0: "raccoon", 1: "rodent"}

    def load_images(self, test_images_dir):
        """Load all pest test images"""
        imgs = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            imgs.extend(glob.glob(os.path.join(test_images_dir, ext)))
        pests = []
        for img_path in imgs:
            img = cv2.imread(img_path)
            if img is not None:
                # Convert to grayscale for consistency
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                pests.append(img)
        print(f"✅ Loaded {len(pests)} grayscale pest images for {self.mode} mode.")
        return pests

    def overlay(self, bg):
        """Overlay pest on background with realistic size"""
        if not self.images:
            return bg, None, None

        pest = random.choice(self.images)
        h, w = pest.shape[:2]

        # Different scale ranges based on pest type (raccoon vs rodent)
        if self.mode == TestMode.RACCOON_ONLY:
            scale = random.uniform(0.3, 0.5)  # Larger for raccoons
        else:
            # 70% chance rodent (small), 30% chance raccoon (large)
            scale = random.uniform(0.15, 0.3) if random.random() < 0.7 else random.uniform(0.3, 0.5)
        
        new_w, new_h = int(w * scale), int(h * scale)
        pest_resized = cv2.resize(pest, (new_w, new_h))

        frame_w, frame_h = self.frame_size

        # Ensure pest fits inside frame
        if new_w >= frame_w or new_h >= frame_h:
            scale = min((frame_w * 0.8) / w, (frame_h * 0.8) / h)
            new_w, new_h = int(w * scale), int(h * scale)
            pest_resized = cv2.resize(pest, (new_w, new_h))

        # Compute valid bounds
        x_min = int(frame_w * 0.1)
        x_max = max(x_min, frame_w - new_w - x_min - 1)
        y_min = int(frame_h * 0.1)
        y_max = max(y_min, frame_h - new_h - y_min - 1)

        # Sample random position safely
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)

        # Roi blending
        roi = bg[y:y+new_h, x:x+new_w]

        if roi.shape[:2] != pest_resized.shape[:2]:
            pest_resized = cv2.resize(pest_resized, (roi.shape[1], roi.shape[0]))

        blend_ratio = 0.8
        blended = cv2.addWeighted(pest_resized, blend_ratio, roi, 1 - blend_ratio, 0)

        bg[y:y+new_h, x:x+new_w] = blended
        
        # Determine class based on size (approximate)
        pest_class = 0 if scale > 0.25 else 1  # 0=raccoon, 1=rodent
        
        return bg, (x, y, x + new_w, y + new_h), pest_class

# === METRICS ===
def iou(box1, box2):
    if not box1 or not box2:
        return 0.0
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2
    xi1, yi1 = max(x11, x12), max(y11, y12)
    xi2, yi2 = min(x21, x22), min(y21, y22)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = ((x21 - x11) * (y21 - y11)) + ((x22 - x12) * (y22 - y12)) - inter
    return inter / (union + 1e-9)

# === DRAWING ===
def draw_info(frame, detections, stats, frame_idx, mode, conf_thres, test_mode):
    for (x1, y1, x2, y2, conf, class_id) in detections:
        # Color code by class and confidence
        if class_id == 0:  # Raccoon
            color = (0, 255, 0) if conf > 0.5 else (0, 200, 255)
            label = f"raccoon {conf:.2f}"
        else:  # Rodent
            color = (255, 0, 0) if conf > 0.5 else (200, 0, 255)
            label = f"rodent {conf:.2f}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    recall = stats["detected"] / max(1, stats["true"])
    precision = stats["detected"] / max(1, stats["detected"] + stats["false_pos"])

    # Color code metrics
    recall_color = (0, 255, 0) if recall > 0.85 else (0, 165, 255) if recall > 0.7 else (0, 0, 255)
    precision_color = (0, 255, 0) if precision > 0.85 else (0, 165, 255) if precision > 0.7 else (0, 0, 255)

    lines = [
        f"Mode: {test_mode.upper()}",
        f"Frame: {frame_idx}/{stats['total']}",
        f"Env: {mode}",
        f"Conf: {conf_thres:.2f}",
        f"True: {stats['true']}",
        f"Detected: {stats['detected']}",
        f"FP: {stats['false_pos']}",
        f"Recall: {recall:.2f}",
        f"Precision: {precision:.2f}"
    ]

    for i, text in enumerate(lines):
        color = (255, 255, 255)
        if "Recall" in text:
            color = recall_color
        elif "Precision" in text:
            color = precision_color
        elif "Mode" in text:
            color = (0, 255, 255)  # Cyan for mode indicator

        cv2.putText(frame, text, (10, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

# === MAIN TEST FUNCTION ===
def run_test(test_mode=TestMode.GENERAL_PEST):
    print(f"Running Synthetic Test - Mode: {test_mode.upper()}")
    
    # Load appropriate model and test images
    model_path = MODEL_PATHS[test_mode]
    test_images_dir = TEST_IMAGES_DIRS[test_mode]
    
    model = YOLO(model_path)
    env = GrayscaleEnvironment()
    overlay = PestOverlay(frame_size=FRAME_SIZE, test_images_dir=test_images_dir, mode=test_mode)

    # Load or synthesize grayscale backgrounds
    bg_images = []
    for folder in ["nightcam_backgrounds", "daycam_backgrounds"]:
        for path in glob.glob(f"{folder}/*.*"):
            img = cv2.imread(path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                bg_images.append(cv2.resize(gray, FRAME_SIZE))
    if not bg_images:
        print("⚠️ No backgrounds found, creating synthetic ones.")
        for _ in range(6):
            base = np.random.randint(20, 80, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            bg_images.append(base)

    stats = {
        "true": 0, "detected": 0, "false_pos": 0, "total": int(DURATION * FPS),
        "raccoon_true": 0, "raccoon_detected": 0, "rodent_true": 0, "rodent_detected": 0
    }

    output_file = f"synthetic_test_{test_mode}.mp4"
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    cv2.namedWindow(f"Pest Detection - {test_mode.upper()}", cv2.WINDOW_NORMAL)

    for i in range(stats["total"]):
        env_mode = random.choices(config.MODES, config.MODE_WEIGHTS)[0]
        bg = random.choice(bg_images).copy()
        bg = env.apply_effect(bg, env_mode)

        # Possibly overlay a pest
        pest_bbox = None
        pest_class = None
        show_pest = random.random() < config.PEST_PROB
        if show_pest:
            bg, pest_bbox, pest_class = overlay.overlay(bg)
            stats["true"] += 1
            if pest_class == 0:
                stats["raccoon_true"] += 1
            else:
                stats["rodent_true"] += 1

        # Detect
        conf_thres = config.CONFIDENCE_THRESHOLDS[env_mode]
        results = model(bg, conf=conf_thres, verbose=False, iou=0.5)
        detections = []
        pest_detected = False

        for box in results[0].boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # In raccoon-only mode, ignore rodent detections
            if test_mode == TestMode.RACCOON_ONLY and class_id == 1:
                continue

            # Ensure realistic detections
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            aspect_ratio = bbox_width / max(1, bbox_height)

            min_area = 1500
            max_area = 70000
            min_aspect = 0.35  
            max_aspect = 2.8

            if (bbox_area >= min_area and bbox_area <= max_area and 
                aspect_ratio >= min_aspect and aspect_ratio <= max_aspect):

                detections.append((x1, y1, x2, y2, conf, class_id))

                if pest_bbox:
                    current_iou = iou((x1, y1, x2, y2), pest_bbox)

                    if current_iou > config.IOU_THRESHOLD and not pest_detected:
                        pest_detected = True
                        stats["detected"] += 1
                        if pest_class == 0:
                            stats["raccoon_detected"] += 1
                        else:
                            stats["rodent_detected"] += 1

        if not show_pest:
            stats["false_pos"] += len(detections)
        elif show_pest and pest_detected:
            # Only count EXTRA detections beyond the correct one
            stats["false_pos"] += max(0, len(detections) - 1)
        elif show_pest and not pest_detected:
            stats["false_pos"] += len(detections)

        frame = draw_info(bg, detections, stats, i + 1, env_mode, conf_thres, test_mode)
        out.write(frame)
        cv2.imshow(f"Pest Detection - {test_mode.upper()}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cv2.destroyAllWindows()

    recall = stats["detected"] / max(1, stats["true"])
    precision = stats["detected"] / max(1, stats["detected"] + stats["false_pos"])
    
    raccoon_recall = stats["raccoon_detected"] / max(1, stats["raccoon_true"])
    rodent_recall = stats["rodent_detected"] / max(1, stats["rodent_true"])

    print(f"\n=== {test_mode.upper()} TEST SUMMARY ===")
    print(f"Frames: {stats['total']}")
    print(f"True Pests: {stats['true']} (Raccoons: {stats['raccoon_true']}, Rodents: {stats['rodent_true']})")
    print(f"Detected: {stats['detected']} (Raccoons: {stats['raccoon_detected']}, Rodents: {stats['rodent_detected']})")
    print(f"False Positives: {stats['false_pos']}")
    print(f"Precision: {precision:.3f}")
    print(f"Overall Recall: {recall:.3f}")
    print(f"Raccoon Recall: {raccoon_recall:.3f}")
    if test_mode == TestMode.GENERAL_PEST:
        print(f"Rodent Recall: {rodent_recall:.3f}")
    print(f"Video saved: {output_file}")
    
    return stats

def run_comparative_test():
    """Run both modes and compare results"""
    print("🚀 STARTING COMPARATIVE TEST")
    print("=" * 50)
    
    run_test(TestMode.RACCOON_ONLY)
    print("\n" + "=" * 50)
    run_test(TestMode.GENERAL_PEST)
    print("\n" + "=" * 50)

if __name__ == "__main__":
    # Run individual test or comparative test
    # run_test(TestMode.RACCOON_ONLY)  # Test one mode
    # run_test(TestMode.GENERAL_PEST)  # Test the other mode
    run_comparative_test()  # Test both and compare