from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=r"C:\Users\PALLAVI\Downloads\Pothole.v1-raw.yolov8\data.yaml",
    epochs=20,
    imgsz=640
)