# Pothole Detection

A computer vision model that detects potholes in road footage, built with YOLO26.
The goal is a system that could power either real-time driver alerts or a map of
road damage for repair crews.

## What it does

The model takes road footage (from a forward-facing or dashcam-style camera) and
draws bounding boxes around potholes it detects, frame by frame.

## Results

Trained on a public pothole dataset, the model reached:
- **mAP@50:** ~0.89
- **Precision:** ~0.87
- **Recall:** ~0.83
- **Inference speed:** ~4.4 ms per frame (well above real-time)

## How it works

- **Model:** YOLO26 (small variant), via the Ultralytics library
- **Training:** Google Colab with a T4 GPU
- **Dataset:** public pothole dataset (Roboflow)
- **Approach:** transfer learning from pre-trained weights, then fine-tuned on potholes

## Running it

```python
from ultralytics import YOLO

model = YOLO("best.pt")          # the trained model
model.predict(source="road.mp4", save=True, conf=0.5)
```

## Project status

Detection on recorded video is working. Current focus is expanding the training
data to improve reliability across different conditions (rain, night, road types).

## Roadmap

- [x] Train model and validate accuracy
- [x] Run detection on recorded video
- [ ] Improve robustness across lighting and weather conditions
- [ ] GPS logging and a cloud map of detected potholes
- [ ] Real-time in-car driver alert
- [ ] Deploy to edge hardware (NVIDIA Jetson)
