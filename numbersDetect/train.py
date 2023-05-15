from ultralytics import YOLO
import time
# Load a model
model = YOLO()


results = model.train(data="numbersDetect\dataNumbers.yaml", epochs=300)  # train the model
success = YOLO("number.pt").export(format="onnx")

