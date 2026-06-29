from ultralytics import YOLO

model = YOLO("best.pt")   # your trained model

model.predict(
    source="road.mp4",    # video or image to run on
    save=True,
    conf=0.5,
)
