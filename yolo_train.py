from ultralytics import YOLO



# Load a model

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)



# Train the model

results = model.train(data="splited", epochs=10, imgsz=180)
