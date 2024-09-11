from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/GC-YOLO.yaml")

result = model.train(data="ultralytics/cfg/datasets/VisDrone.yaml",epochs=500,imgsz=640)