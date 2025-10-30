import os
import cv2
import random
import numpy as np
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/detect/train4/weights/best.pt"
TEST_IMAGES_DIR = "augmentation_dataset/test/images"
OUTPUT_VIDEO = "synthetic_realistic_pest_feed_v2.mp4"

# === PARAMETERS ===
FRAME_SIZE = (640, 640)
FPS = 10
DURATION = 60
PEST_PROB = 0.05  # probability of pest per frame
CONF_THRESHOLD = 0.60
AUGMENT_PROB = 0.4

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === LOAD PEST IMAGES ===
pest_images = sorted([
    os.path.join(TEST_IMAGES_DIR, f)
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

# === BACKGROUND ===
def generate_background(prev_bg=None):
    """Simulate camera motion background."""
    if prev_bg is None:
        bg = np.random.randint(100, 160, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    else:
        bg = prev_bg.copy()
        shift_x, shift_y = random.randint(-2, 2), random.randint(-2, 2)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        bg = cv2.warpAffine(bg, M, (FRAME_SIZE[0], FRAME_SIZE[1]))
        bg = cv2.convertScaleAbs(bg, alpha=1 + (random.random() - 0.5) * 0.05, beta=random.randint(-5, 5))
    return bg

# === AUGMENT ===
def random_augment(img):
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

# === OVERLAY PEST ===
def overlay_pest(background, pest_img):
    pest_img = cv2.resize(pest_img, (random.randint(80, 200), random.randint(80, 200)))
    h, w, _ = pest_img.shape
    x, y = random.randint(0, FRAME_SIZE[0] - w), random.randint(0, FRAME_SIZE[1] - h)
    roi = background[y:y+h, x:x+w]
    pest_mask = cv2.cvtColor(pest_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(pest_mask, 1, 255, cv2.THRESH_BINARY)
    pest_img = cv2.bitwise_and(pest_img, pest_img, mask=mask)
    bg_mask = cv2.bitwise_not(mask)
    roi_bg = cv2.bitwise_and(roi, roi, mask=bg_mask)
    blended = cv2.add(roi_bg, pest_img)
    background[y:y+h, x:x+w] = blended
    return background

# === DE-DUPLICATE DETECTIONS ===
def filter_overlapping_boxes(boxes, threshold=0.4):
    """Simple non-max suppression-like filter."""
    filtered = []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # sort by confidence
    while boxes:
        best = boxes.pop(0)
        filtered.append(best)
        boxes = [
            b for b in boxes
            if iou(best, b) < threshold
        ]
    return filtered

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# === VIDEO ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)
cv2.namedWindow("Synthetic Pest Feed", cv2.WINDOW_NORMAL)

# === STATS ===
true_pest_count = 0
detected_pest_count = 0
false_positive_count = 0
prev_detection = False
frame_count = int(DURATION * FPS)
background = None

for i in range(frame_count):
    background = generate_background(background)
    show_pest = random.random() < PEST_PROB

    if show_pest and pest_images:
        pest_img = cv2.imread(random.choice(pest_images))
        background = overlay_pest(background, pest_img)
        true_pest_count += 1

    if random.random() < AUGMENT_PROB:
        background = random_augment(background)

    # === RUN DETECTION ===
    results = model(background, verbose=False)
    boxes = results[0].boxes
    detected = []

    for box in boxes:
        conf = float(box.conf[0])
        if conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected.append([x1, y1, x2, y2, conf])

    detected = filter_overlapping_boxes(detected)
    has_detection = len(detected) > 0

    # === TEMPORAL FILTER ===
    if has_detection and prev_detection:
        # Only count confirmed detection if persistent
        detected_pest_count += 1
    elif not show_pest and has_detection:
        false_positive_count += 1

    prev_detection = has_detection

    # === DRAW ===
    for (x1, y1, x2, y2, conf) in detected:
        cv2.rectangle(background, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(background, f"Pest {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # === METRICS ===
    recall = detected_pest_count / (true_pest_count + 1e-6)
    precision = detected_pest_count / (detected_pest_count + false_positive_count + 1e-6)

    cv2.putText(background, f"Frame {i+1}/{frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(background, f"True Pests: {true_pest_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(background, f"Detected: {detected_pest_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(background, f"False Positives: {false_positive_count}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(background, f"Recall: {recall:.2f}  Precision: {precision:.2f}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out.write(background)
    cv2.imshow("Synthetic Pest Feed", background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

# === SUMMARY ===
print("\n=== Synthetic Realistic Pest Feed (v2) Completed ===")
print(f"Frames: {frame_count}")
print(f"True Pests: {true_pest_count}")
print(f"Detected (filtered): {detected_pest_count}")
print(f"False Positives: {false_positive_count}")
recall = detected_pest_count / (true_pest_count + 1e-6)
precision = detected_pest_count / (detected_pest_count + false_positive_count + 1e-6)
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Output saved to: {OUTPUT_VIDEO}")
