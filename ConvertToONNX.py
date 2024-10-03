from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("runs/train/FTCYolo45/weights/best.pt")

# Export the model to ONNX format
model.export(format="onnx", opset=12)  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("runs/train/FTCYolo45/weights/best.onnx")
print("loaded onnx model")