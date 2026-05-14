"""Preprocess VisDrone labels: merge pedestrian+people -> human(0), car+van -> car(1)."""
import os, glob, sys, random, cv2, matplotlib.pyplot as plt
from collections import Counter

# Original VisDrone class IDs -> merged IDs (0=human, 1=car)
REMAP = {0: 0, 1: 0, 3: 1, 4: 1}
NAMES = {0: "human", 1: "car"}

def remap_labels(data_root):
    """Filter and remap label files in-place."""
    splits = ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]
    stats = Counter()
    for split in splits:
        label_dir = os.path.join(data_root, split, "labels")
        if not os.path.exists(label_dir):
            print(f"Skipping {split} (not found)")
            continue
        files = glob.glob(os.path.join(label_dir, "*.txt"))
        for f in files:
            with open(f, "r") as fh:
                lines = fh.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in REMAP:
                    parts[0] = str(REMAP[cls])
                    new_lines.append(" ".join(parts) + "\n")
                    stats[NAMES[REMAP[cls]]] += 1
            with open(f, "w") as fh:
                fh.writelines(new_lines)
        print(f"Processed {len(files)} files in {split}")
    print(f"\nClass distribution: {dict(stats)}")

def visualize_samples(data_root, n=6):
    """Show sample images with bounding boxes."""
    img_dir = os.path.join(data_root, "VisDrone2019-DET-train", "images")
    lbl_dir = os.path.join(data_root, "VisDrone2019-DET-train", "labels")
    colors = [(0,255,0), (255,0,0)]
    imgs = random.sample(glob.glob(os.path.join(img_dir, "*.jpg")), min(n, len(os.listdir(img_dir))))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, img_path in zip(axes.flat, imgs):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lbl_path = os.path.join(lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if os.path.exists(lbl_path):
            for line in open(lbl_path):
                parts = line.strip().split()
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1, y1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
                x2, y2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
                cv2.rectangle(img, (x1,y1), (x2,y2), colors[cls], 2)
                cv2.putText(img, NAMES[cls], (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path), fontsize=8)
        ax.axis("off")
    plt.suptitle("Sample Training Images with Bounding Boxes", fontsize=14)
    plt.tight_layout()
    plt.savefig("sample_visualizations.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved sample_visualizations.png")

if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "VisDrone_Dataset"
    print("=== Remapping labels ===")
    remap_labels(data_root)
    print("\n=== Visualizing samples ===")
    visualize_samples(data_root)
