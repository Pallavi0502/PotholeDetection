from ultralytics import YOLO

# Load trained model
model = YOLO(r"runs\detect\train\weights\best.pt")

# Ask for image
image_path = input("Enter image path: ")

# Predict
results = model.predict(
    source=image_path,
    save=True
)

# Get first result
result = results[0]

# Number of detected objects
count = len(result.boxes)

print("Prediction completed!")
print("Number of potholes detected:", count)

# Display confidence scores
for i, box in enumerate(result.boxes):
    confidence = float(box.conf[0])
    print(f"Pothole {i + 1}: Confidence = {confidence:.2f}")