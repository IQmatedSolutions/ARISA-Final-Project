from ultralytics import YOLO
import os
import shutil

model = YOLO("yolov5nu.pt")

# Train the model with use of data augmentation
results = model.train(data="dataset/processed/data.yaml", epochs=100, imgsz=640, device=0, freeze = 24, patience = 10)

# Create "my_model" folder to store model weights and train results
os.mkdir("/models/my_model")
shutil.copyfile("runs/detect/train/weights/best.pt", "/models/my_model/ARISA.pt")

# Export model to ncnn version so it can be used by Raspberry Pi
model = YOLO("/models/my_model/ARISA.pt")
model.export(format="ncnn")
