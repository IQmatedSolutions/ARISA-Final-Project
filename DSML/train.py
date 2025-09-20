from ultralytics import YOLO
from ultralytics import settings

# Update a setting
settings.update({"mlflow": True})

model = YOLO("yolov5nu.pt")

# Train the model with use of data augmentation
results = model.train(data="dataset/processed/data.yaml", epochs=30, imgsz=640, device=0, freeze = 10)
