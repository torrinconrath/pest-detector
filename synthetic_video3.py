"""
Test 3: Adaptive Synthetic Pest Feed
------------------------------------
This version blends pest images with real nightcam/daycam backgrounds,
and automatically adjusts lighting, brightness, and contrast
for more realistic and detectable scenes.
"""

import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
import glob

# === CONFIG ===
MODEL_PATH = "runs/detect/train/weights/best.pt"
TEST_IMAGES_DIR = "rebalanced_dataset/test/images"

# === CHOOSE ONE BACKGROUND SET (DAY OR NIGHT) ===
mode = random.choice(["day", "night"])  # or manually set: mode = "day"
mode = "night"

if mode == "day":
    BACKGROUND_IMAGES = glob.glob("daycam_backgrounds/*.*")
else:
    BACKGROUND_IMAGES = glob.glob("nightcam_backgrounds/*.*")

if not BACKGROUND_IMAGES:
    raise FileNotFoundError(f"No background images found for mode '{mode}'!")

print(f"Running adaptive feed in '{mode.upper()}' mode with {len(BACKGROUND_IMAGES)} backgrounds.")

# === LOAD BACKGROUNDS AS GRAYSCALE (3-channel) ===
backgrounds = []
for bg_path in BACKGROUND_IMAGES:
    img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    # Keep 3 channels for YOLO inference compatibility
    img_gray_3ch = cv2.merge([img, img, img])
    backgrounds.append(img_gray_3ch)

if not backgrounds:
    raise FileNotFoundError("No valid grayscale background images could be loaded!")

print(f"✅ Loaded {len(backgrounds)} grayscale background images.")

OUTPUT_VIDEO = "synthetic_pest_feed_adaptive_v3.mp4"

# === PARAMETERS ===
FRAME_SIZE = (640, 640)
FPS = 10
DURATION = 60
PEST_PROB = 0.05
CONF_THRESHOLD = 0.5
AUGMENT_PROB = 0.25
BRIGHTEN_PEST = True

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === LOAD TEST PEST IMAGES ===
pest_images = sorted([
    os.path.join(TEST_IMAGES_DIR, f)
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

# === UTILS ===

def match_lighting(img, target_brightness=160):
    """Normalize brightness to target value."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean == 0:
        return img
    scale = target_brightness / mean
    img = np.clip(img * scale, 0, 255).astype(np.uint8)
    return img

def adjust_gamma(image, gamma=1.5):
    """Apply gamma correction."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def random_augment(img):
    """Simulate camera/environment noise."""
    if random.random() < 0.3:
        angle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    return img

def overlay_pest(background, pest_img):
    """Overlay pest onto the background at random location."""
    pest_img = cv2.resize(pest_img, (random.randint(150, 400), random.randint(150, 400)))
    h, w, _ = pest_img.shape
    x, y = random.randint(0, FRAME_SIZE[0] - w), random.randint(0, FRAME_SIZE[1] - h)

    pest_gray = cv2.cvtColor(pest_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(pest_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi = background[y:y+h, x:x+w]
    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
    pest_part = cv2.bitwise_and(pest_img, pest_img, mask=mask)
    combined = cv2.add(bg_part, pest_part)
    background[y:y+h, x:x+w] = combined

    return background

def overlay_pest_with_brightness(background, pest_img):
    """Brighten pest before overlay for visibility in dark scenes."""
    pest_img = cv2.convertScaleAbs(pest_img, alpha=1.4, beta=10)
    return overlay_pest(background, pest_img)

def normalize_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.merge([eq, eq, eq])

# === VIDEO ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)
cv2.namedWindow("Synthetic Pest Feed", cv2.WINDOW_NORMAL)

# === STATS ===
true_pest_count = 0
detected_pest_count = 0
false_positive_count = 0
frame_count = int(DURATION * FPS)

# === MAIN LOOP ===
for i in range(frame_count):
    bg = random.choice(backgrounds)
    bg = cv2.resize(bg, FRAME_SIZE)

    # Apply lighting adjustment (simulate night/day conditions)
    if random.random() < 0.5:
        bg = adjust_gamma(bg, gamma=random.uniform(1.2, 1.8))
    else:
        bg = match_lighting(bg, target_brightness=random.randint(140, 180))

    show_pest = random.random() < PEST_PROB

    if show_pest and pest_images:
        pest_img = cv2.imread(random.choice(pest_images))
        pest_img = match_lighting(pest_img, target_brightness=random.randint(130, 180))
        if BRIGHTEN_PEST:
            bg = overlay_pest_with_brightness(bg, pest_img)
        else:
            bg = overlay_pest(bg, pest_img)
        true_pest_count += 1

    if random.random() < AUGMENT_PROB:
        bg = random_augment(bg)

    # === YOLO INFERENCE ===
    bg = normalize_lighting(bg)
    results = model(bg, verbose=False)
    boxes = results[0].boxes
    detections = []

    for box in boxes:
        conf = float(box.conf[0])
        if conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, conf))

    if show_pest and len(detections) > 0:
        detected_pest_count += 1
    elif not show_pest and len(detections) > 0:
        false_positive_count += 1

    # === DRAW ===
    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(bg, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(bg, f"Pest {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    recall = detected_pest_count / (true_pest_count + 1e-6)
    precision = detected_pest_count / (detected_pest_count + false_positive_count + 1e-6)

    cv2.putText(bg, f"Frame {i+1}/{frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bg, f"True Pests: {true_pest_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(bg, f"Detected: {detected_pest_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(bg, f"False Positives: {false_positive_count}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(bg, f"Recall: {recall:.2f}  Precision: {precision:.2f}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out.write(bg)
    cv2.imshow("Synthetic Pest Feed", bg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

# === SUMMARY ===
print("\n=== Adaptive Synthetic Pest Feed (Test 3) Completed ===")
print(f"Frames: {frame_count}")
print(f"True Pests: {true_pest_count}")
print(f"Detected: {detected_pest_count}")
print(f"False Positives: {false_positive_count}")
recall = detected_pest_count / (true_pest_count + 1e-6)
precision = detected_pest_count / (detected_pest_count + false_positive_count + 1e-6)
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Output saved to: {OUTPUT_VIDEO}")
