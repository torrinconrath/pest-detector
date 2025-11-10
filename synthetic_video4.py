"""
Test 4: Real-World Pest Detection Simulation (FINAL FIXED VERSION)
------------------------------------------------------------------
Uses only basic OpenCV operations - no albumentations dependency issues.
"""

import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
import glob
from pathlib import Path

# === CONFIG ===
MODEL_PATH = "runs/detect/train/weights/best.pt"
TEST_IMAGES_DIR = "rebalanced_dataset/test/images"

# Frame size 
FRAME_SIZE = (640, 640)
FPS = 8
DURATION = 60
OUTPUT_VIDEO = "real_world_pest_detection_v4_final.mp4"

# === SIMULATION PARAMETERS ===
class RealWorldConfig:
    MODES = ["clear_night", "foggy_night", "rainy_night", "clear_dusk", "overcast"]
    MODE_WEIGHTS = [0.3, 0.2, 0.2, 0.15, 0.15]

    PEST_PROBABILITIES = {
        "raccoon": 0.06,  # Higher probability for testing
        "rodent": 0.08    
    }

    CONFIDENCE_THRESHOLDS = {
        "clear_night": 0.3,  # Much lower thresholds
        "foggy_night": 0.25,
        "rainy_night": 0.2,
        "clear_dusk": 0.3,
        "overcast": 0.35
    }

config = RealWorldConfig()

# === PURE OPENCV ENVIRONMENT SIMULATION ===
class OpenCVEnvironmentalSimulator:
    def apply_environment(self, image, environment):
        """Apply environmental effects using only OpenCV"""
        img = image.copy().astype(np.float32)
        
        if environment == "clear_night":
            # Darken + noise
            img = img * 0.4 + np.random.normal(0, 8, img.shape)
            
        elif environment == "foggy_night":
            # Darken + blur + slight brightening for fog
            img = img * 0.5
            img = cv2.GaussianBlur(img.astype(np.uint8), (7, 7), 2).astype(np.float32)
            img = img * 1.1
            
        elif environment == "rainy_night":
            # Darken + noise + motion blur
            img = img * 0.6
            # Add rain streaks (vertical lines)
            for _ in range(20):
                x = random.randint(0, img.shape[1]-1)
                width = random.randint(1, 3)
                img[:, x:x+width] += random.randint(20, 40)
            # Motion blur
            kernel = np.zeros((5, 5))
            kernel[2, :] = 1/5  # Horizontal motion blur
            img = cv2.filter2D(img.astype(np.uint8), -1, kernel)
            
        elif environment == "clear_dusk":
            # Warm tones (boost red, reduce blue)
            img[:, :, 0] = img[:, :, 0] * 0.9  # Reduce blue
            img[:, :, 2] = img[:, :, 2] * 1.2  # Boost red
            img = img * 0.8  # Slightly darker
            
        elif environment == "overcast":
            # Desaturate and flatten contrast
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            img = cv2.merge([gray, gray, gray]).astype(np.float32)
            img = img * 0.9
            
        # Clip and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_environment_description(self, environment):
        descriptions = {
            "clear_night": "Clear Night",
            "foggy_night": "Foggy Night", 
            "rainy_night": "Rainy Night",
            "clear_dusk": "Clear Dusk",
            "overcast": "Overcast Day"
        }
        return descriptions.get(environment, "Unknown")

# === SIMPLE BUT RELIABLE PEST OVERLAY ===
class SimplePestOverlay:
    def __init__(self, frame_size):
        self.frame_size = frame_size
        self.pest_images = self.load_pest_images()
        
    def load_pest_images(self):
        """Load and properly classify pest images"""
        pest_images = {"raccoon": [], "rodent": []}
        
        # Get all test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
        
        print(f"Found {len(test_images)} test images")
        
        # Use actual filenames for classification
        for img_path in test_images:
            img = cv2.imread(img_path)
            if img is not None:
                filename = os.path.basename(img_path).lower()
                if 'raccoon' in filename or 'coon' in filename:
                    pest_images["raccoon"].append(img)
                else:
                    pest_images["rodent"].append(img)
        
        # If classification failed, use size-based fallback
        if len(pest_images["raccoon"]) == 0 or len(pest_images["rodent"]) == 0:
            print("Using size-based classification fallback...")
            pest_images = {"raccoon": [], "rodent": []}
            for img_path in test_images:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    # Raccoons are generally larger in most datasets
                    if h > 400 or w > 400:
                        pest_images["raccoon"].append(img)
                    else:
                        pest_images["rodent"].append(img)
        
        print(f"✅ Loaded: {len(pest_images['raccoon'])} raccoons, {len(pest_images['rodent'])} rodents")
        return pest_images
    
    def simple_overlay(self, background, pest_image, pest_type, environment):
        """Very simple but reliable overlay"""
        try:
            # Scale pest appropriately
            if pest_type == "raccoon":
                scale = random.uniform(0.25, 0.4)
            else:
                scale = random.uniform(0.2, 0.35)
                
            h, w = pest_image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Ensure minimum size
            new_w, new_h = max(80, new_w), max(80, new_h)
            
            # Resize pest
            pest_resized = cv2.resize(pest_image, (new_w, new_h))
            
            # Choose position
            frame_w, frame_h = self.frame_size
            x = random.randint(0, max(1, frame_w - new_w))
            y = random.randint(0, max(1, frame_h - new_h))
            
            # SIMPLE OVERLAY: Just paste the pest (no complex blending)
            # This ensures the pest is clearly visible for testing
            background[y:y+new_h, x:x+new_w] = pest_resized
            
            return background, (x, y, x + new_w, y + new_h), pest_type
            
        except Exception as e:
            print(f"⚠️  Overlay failed: {e}")
            return background, None, pest_type

# === UTILITY FUNCTIONS ===
def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    if box1 is None or box2 is None:
        return 0.0

    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2

    xi1 = max(x11, x12)
    yi1 = max(y11, y12)
    xi2 = min(x21, x22)
    yi2 = min(y21, y22)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)

    union_area = box1_area + box2_area - inter_area + 1e-9
    return inter_area / union_area

def draw_detection_results(image, detections, pest_detected, frame_idx, stats, env_description, conf_threshold):
    """Draw detection results"""
    # Draw all detections
    for (x1, y1, x2, y2, conf, class_name) in detections:
        color = (0, 255, 0)  # Green for all detections
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Calculate metrics
    recall = stats["detected_pests"] / max(1, stats["true_pests"])
    precision = stats["detected_pests"] / max(1, stats["detected_pests"] + stats["false_positives"])

    # Draw info
    info_lines = [
        f"Frame: {frame_idx + 1}/{stats['total_frames']}",
        f"Env: {env_description}",
        f"Conf: {conf_threshold:.2f}",
        f"True: {stats['true_pests']}",
        f"Detected: {stats['detected_pests']}",
        f"FP: {stats['false_positives']}",
        f"Recall: {recall:.2f}",
        f"Precision: {precision:.2f}"
    ]

    for i, line in enumerate(info_lines):
        y_pos = 20 + i * 20
        color = (255, 255, 255)
        if "Recall" in line:
            color = (0, 255, 0) if recall > 0.5 else (0, 0, 255)
        elif "Precision" in line:
            color = (0, 255, 0) if precision > 0.5 else (0, 0, 255)
        cv2.putText(image, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

# === MAIN SIMULATION ===
def run_real_world_simulation():
    print("🚀 Starting FINAL Real-World Pest Detection Simulation")
    print("=" * 60)
    print("Using pure OpenCV - no albumentations dependencies")

    # Initialize
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    env_simulator = OpenCVEnvironmentalSimulator()
    pest_overlay = SimplePestOverlay(frame_size=FRAME_SIZE)

    # Load backgrounds
    backgrounds = []
    night_bgs = glob.glob("nightcam_backgrounds/*.*")
    day_bgs = glob.glob("daycam_backgrounds/*.*")
    
    all_bgs = night_bgs + day_bgs
    if all_bgs:
        for bg_path in all_bgs:
            img = cv2.imread(bg_path)
            if img is not None:
                backgrounds.append(img)
        print(f"✅ Loaded {len(backgrounds)} background images")
    else:
        # Create simple synthetic backgrounds
        print("⚠️  No background images found, creating synthetic ones...")
        for i in range(3):
            # Night background
            bg = np.random.randint(0, 60, (480, 640, 3), dtype=np.uint8)
            backgrounds.append(bg)
            # Day background  
            bg = np.random.randint(100, 180, (480, 640, 3), dtype=np.uint8)
            backgrounds.append(bg)

    # Video setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)

    # Statistics
    stats = {
        "total_frames": int(DURATION * FPS),
        "true_pests": 0,
        "detected_pests": 0, 
        "false_positives": 0,
    }

    print(f"🎬 Testing {stats['total_frames']} frames")
    print("📊 Monitoring detection performance...")
    
    cv2.namedWindow("Pest Detection Test", cv2.WINDOW_NORMAL)

    for frame_idx in range(stats["total_frames"]):
        # Select environment
        current_env = random.choices(config.MODES, weights=config.MODE_WEIGHTS)[0]
        env_description = env_simulator.get_environment_description(current_env)
        
        # Get and resize background
        if backgrounds:
            bg = random.choice(backgrounds)
            bg = cv2.resize(bg, FRAME_SIZE)
        else:
            bg = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        
        # Apply environment
        bg = env_simulator.apply_environment(bg, current_env)
        
        # Add pest with higher probability
        show_pest = False
        pest_bbox = None
        pest_type = None
        
        for p_type, prob in config.PEST_PROBABILITIES.items():
            if random.random() < prob and pest_overlay.pest_images[p_type]:
                show_pest = True
                pest_type = p_type
                pest_img = random.choice(pest_overlay.pest_images[p_type])
                bg, pest_bbox, _ = pest_overlay.simple_overlay(bg.copy(), pest_img, pest_type, current_env)
                
                if pest_bbox:
                    stats["true_pests"] += 1
                    if frame_idx % 50 == 0:  # Print every 50 frames
                        print(f"Frame {frame_idx}: Added {pest_type} at {pest_bbox}")
                break

        # Run detection with very low confidence threshold
        conf_threshold = config.CONFIDENCE_THRESHOLDS.get(current_env, 0.2)
        results = model(bg, conf=conf_threshold, verbose=False)
        
        # Process results
        detections = []
        pest_detected = False
        
        for box in results[0].boxes:
            try:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = "raccoon" if class_id == 0 else "rodent"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, conf, class_name))
                
                # Check if matches our pest
                if show_pest and pest_bbox and not pest_detected:
                    iou = calculate_iou((x1, y1, x2, y2), pest_bbox)
                    if iou > 0.2:  # Lower IoU threshold
                        pest_detected = True
                        stats["detected_pests"] += 1
                        if frame_idx % 50 == 0:
                            print(f"Frame {frame_idx}: ✅ Detected {pest_type} with IoU {iou:.2f}")
            except Exception as e:
                continue

        # Count false positives
        if not show_pest and detections:
            stats["false_positives"] += len(detections)
            if frame_idx % 50 == 0 and detections:
                print(f"Frame {frame_idx}: ⚠️  False positives: {len(detections)}")
        elif show_pest and not pest_detected and detections:
            stats["false_positives"] += len(detections)
        elif show_pest and pest_detected and len(detections) > 1:
            stats["false_positives"] += len(detections) - 1

        # Draw and display
        display_img = draw_detection_results(bg, detections, pest_detected, frame_idx, stats, env_description, conf_threshold)
        out.write(display_img)
        cv2.imshow("Pest Detection Test", display_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted simulation")
            break

    # Cleanup and report
    out.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*50)
    print("📊 FINAL RESULTS")
    print("="*50)
    
    recall = stats["detected_pests"] / max(1, stats["true_pests"])
    precision = stats["detected_pests"] / max(1, stats["detected_pests"] + stats["false_positives"])
    
    print(f"Total Frames: {stats['total_frames']}")
    print(f"True Pests: {stats['true_pests']}")
    print(f"Detected Pests: {stats['detected_pests']}")
    print(f"False Positives: {stats['false_positives']}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    
    if recall > 0.7:
        print("🎉 EXCELLENT detection performance!")
    elif recall > 0.5:
        print("✅ GOOD detection performance!")
    elif recall > 0.3:
        print("⚠️  MODERATE detection performance")
    else:
        print("❌ POOR detection performance")
    
    print(f"🎥 Video saved: {OUTPUT_VIDEO}")

# Run the simulation
if __name__ == "__main__":
    run_real_world_simulation()