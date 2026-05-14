# Drone Human & Car Detection + Tracking (YOLO26)

Aerial object detection and tracking using **YOLO26** (latest Ultralytics) on the VisDrone dataset. Detects humans (pedestrian/people) and cars (car/van), counts humans, and tracks objects using BoT-SORT.

## Setup (Google Colab)

### 1. Clone & Install
```bash
git clone https://github.com/taufiktrf/visdrone-human-car-detection-tracking.git
cd visdrone-human-car-detection-tracking
pip install ultralytics opencv-python matplotlib
```

### 2. Download & Extract Dataset
```bash
curl -L -o visdrone-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/banuprasadb/visdrone-dataset
unzip -q visdrone-dataset.zip
mv visdrone.yaml VisDrone_Dataset/
```

### 3. Preprocess (remap labels to 2 classes + visualize)
```bash
python preprocess.py
```

### 4. Train
```bash
python train.py
```

### 5. Detect (on test images)
```bash
python detect.py
```

### 6. Track (on video)
Place a drone video in the project folder, then:
```bash
python track.py test_video.mp4
```

### 7. Evaluate
```bash
python evaluate.py
```

## Output
- `output/detections/` — detected images with bounding boxes + human count
- `output/detection_results.png` — grid visualization
- `output/tracked_output.mp4` — tracked video with trails
- `output/evaluation_plots.png` — training curves and confusion matrix
