"""Train YOLO26 on VisDrone (filtered for human & car detection)."""
import shutil, glob
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.train(
    data="VisDrone_Dataset/visdrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    device=0,
    project="runs",
    name="visdrone_yolo26",
    exist_ok=True,
)

# Copy checkpoints to root
for name in ["best.pt", "last.pt"]:
    src = glob.glob(f"runs/**/weights/{name}", recursive=True)
    if src:
        shutil.copy2(src[0], name)
        print(f"Copied {src[0]} -> {name}")
