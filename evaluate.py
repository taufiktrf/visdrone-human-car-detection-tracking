"""Evaluate trained model and visualize metrics."""
import sys, os, glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_yaml():
    for f in ["VisDrone_Dataset/visdrone.yaml", "data/visdrone.yaml"]:
        if os.path.isfile(f):
            return f
    return "visdrone.yaml"

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "best.pt"

model = YOLO(MODEL_PATH)

# Run validation
metrics = model.val(data=find_yaml(), split="test", conf=0.3, device=0)
print(f"\nmAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

# Display training plots if available
os.makedirs("output", exist_ok=True)
plot_dirs = glob.glob("runs/**/visdrone_yolo26", recursive=True)
plot_dir = plot_dirs[0] if plot_dirs else "runs"
plot_files = ["results.png", "confusion_matrix.png", "P_curve.png", "R_curve.png"]
found = [os.path.join(plot_dir, f) for f in plot_files if os.path.exists(os.path.join(plot_dir, f))]

if found:
    fig, axes = plt.subplots(1, len(found), figsize=(6*len(found), 5))
    if len(found) == 1:
        axes = [axes]
    for ax, path in zip(axes, found):
        ax.imshow(mpimg.imread(path))
        ax.set_title(os.path.basename(path))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved output/evaluation_plots.png")
