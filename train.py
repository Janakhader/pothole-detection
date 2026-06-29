from ultralytics import YOLO

# Start from pre-trained weights and fine-tune on potholes
model = YOLO("yolo26s.pt")

model.train(
    data="data.yaml",   # path to your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0,           # use the GPU
)
