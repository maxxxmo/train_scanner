from ultralytics import YOLO

def export_model(model_path="./runs/detect/runs/train/Train_scanner4/weights/best.pt"):
    model = YOLO(model_path)
    print(f"conversion of {model_path} ongoing...")
    # Quantization to int8 with data.yaml for calibration
    # path = model.export(format="onnx", int8=True, imgsz=640, opset=12, data="data.yaml") # switching to openVINO for better int8 support
    path = model.export(format="openvino", int8=True, imgsz=640, opset=12, data="data.yaml") # Quantization to int8 with data.yaml for calibration
if __name__ == "__main__":
    export_model()