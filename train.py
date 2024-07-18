from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # build a new model from YAML

# Train the model
# results = model.train(data="data.yaml", epochs=1, imgsz=640)
model.predict("image.jpg", save=True, imgsz=640)

