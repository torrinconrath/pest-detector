import os
import cv2
import random
import numpy as np
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/detect/train4/weights/best.pt"
TEST_IMAGES_DIR = "augmentation_dataset/test/images"
TEST_LABELS_DIR = "augmentation_dataset/test/labels"
OUTPUT_VIDEO = "synthetic_anomaly_test.mp4"

# === PARAMETERS ===
FRAME_SIZE = (640, 640)
FPS = 10
DURATION = 60  # seconds
AUGMENT_PROB = 0.3  # chance to apply rotation, blur, etc.
PEST_PROB = 0.05    # 5% chance that a pest (real image) appears

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === LOAD TEST IMAGES ===
image_files = sorted([
    os.path.join(TEST_IMAGES_DIR, f)
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
])

# === UTILS ===
def random_augment(img):
    """Apply random augmentations to simulate camera/environment noise."""
    if random.random() < 0.3:
        angle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    if random.random() < 0.3:
        alpha = 1 + (random.random() - 0.5)
        img = np.clip(cv2.convertScaleAbs(img, alpha=alpha, beta=0), 0, 255)
    return img

# === VIDEO SETUP ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)

window_name = "Synthetic Pest Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# === STATS ===
true_pest_count = 0
detected_pest_count = 0
frame_count = int(DURATION * FPS)

for i in range(frame_count):
    # Decide if frame should contain a pest
    show_pest = random.random() < PEST_PROB

    if show_pest and image_files:
        img_path = random.choice(image_files)
        img = cv2.imread(img_path)
        true_pest_count += 1
    else:
        # Gray background frame
        img = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 180, dtype=np.uint8)

    # Resize and possibly augment
    img = cv2.resize(img, FRAME_SIZE)
    if random.random() < AUGMENT_PROB:
        img = random_augment(img)

    # Run YOLO inference
    results = model(img, verbose=False)
    boxes = results[0].boxes

    # Draw bounding boxes
    for box in boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf > 0.5:
            detected_pest_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Pest {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Overlay status info
    cv2.putText(img, f"Frame: {i+1}/{frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"True Pests: {true_pest_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Detected: {detected_pest_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    out.write(img)
    cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

print(f"\n=== Synthetic Pest Video Completed ===")
print(f"Total Frames: {frame_count}")
print(f"True Pests (Inserted): {true_pest_count}")
print(f"Pests Detected by Model: {detected_pest_count}")
print(f"Output saved to: {OUTPUT_VIDEO}")
