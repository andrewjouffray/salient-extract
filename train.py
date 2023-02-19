from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="data.yaml", epochs=50, imgsz=640, patience=10, batch=8, device=0)