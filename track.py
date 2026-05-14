"""Object tracking on video using BoT-SORT with trained YOLO26 model."""
import sys, os, cv2
from ultralytics import YOLO
from collections import defaultdict

# Single arg = video path, two args = model + video
if len(sys.argv) == 2:
    MODEL_PATH, VIDEO_PATH = "best.pt", sys.argv[1]
elif len(sys.argv) >= 3:
    MODEL_PATH, VIDEO_PATH = sys.argv[1], sys.argv[2]
else:
    MODEL_PATH, VIDEO_PATH = "best.pt", "test_video.mp4"

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = {0: "human", 1: "car"}
COLORS = {0: (0,255,0), 1: (255,0,0)}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
w, h = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out_path = os.path.join(OUT_DIR, "tracked_output.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

track_history = defaultdict(list)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.3, device=0, verbose=False)
    human_count = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box, track_id, cls_id in zip(boxes.xyxy, boxes.id.int(), boxes.cls.int()):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            cls = int(cls_id)
            color = COLORS[cls]
            if cls == 0:
                human_count += 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{tid} {NAMES[cls]}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Track trail
            cx, cy = (x1+x2)//2, (y1+y2)//2
            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)
            pts = track_history[tid]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

    cv2.putText(frame, f"Humans: {human_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    writer.write(frame)

cap.release()
writer.release()
print(f"Tracked {frame_count} frames -> {out_path}")
