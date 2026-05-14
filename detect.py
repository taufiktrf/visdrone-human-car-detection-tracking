"""Detect humans & cars on images, display bounding boxes + human count."""
import sys, os, glob, cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def find_images():
    for d in ["VisDrone_Dataset/VisDrone2019-DET-test-dev/images",
              "data/VisDrone2019-DET-test-dev/images"]:
        if os.path.isdir(d):
            return d
    return "."

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
IMG_DIR = sys.argv[2] if len(sys.argv) > 2 else find_images()
OUT_DIR = "output/detections"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = {0: "human", 1: "car"}
COLORS = {0: (0,255,0), 1: (255,0,0)}

model = YOLO(MODEL_PATH)
images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))[:20]

fig, axes = plt.subplots(4, 5, figsize=(25, 16))
for ax, img_path in zip(axes.flat, images):
    results = model.predict(img_path, conf=0.3, device=0, verbose=False)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    human_count = 0
    for box in results[0].boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        if cls == 0:
            human_count += 1
        cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[cls], 2)
        cv2.putText(img, f"{NAMES[cls]} {conf:.2f}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[cls], 1)
    cv2.putText(img, f"Humans: {human_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ax.imshow(img)
    ax.set_title(f"Humans: {human_count}", fontsize=9)
    ax.axis("off")

plt.suptitle("Detection Results with Human Count", fontsize=16)
plt.tight_layout()
plt.savefig("output/detection_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {len(images)} detection images to {OUT_DIR}/")
