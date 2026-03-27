from ultralytics import YOLO
import os

def test_openvino_performance(model_dir, data_yaml):
    '''Test the performance of an OpenVINO INT8 model on the validation dataset.
    '''
    # Load the OpenVINO INT8 model
    if not os.path.isdir(model_dir):
        print(f"Error: {model_dir} doesnt exist")
        return

    print(f"--- Loading OpenVINO INT8 model: {model_dir} ---")
    model = YOLO(model_dir, task='detect')

    # Validation process to get metrics and inference time

    results = model.val(
        data=data_yaml, 
        imgsz=640, 
        split='val', 
        plots=False, 
        save_json=False
    )

    map50 = results.box.map50  

    # results.speed is : {'preprocess': x, 'inference': y, 'postprocess': z}
    avg_inference_time = results.speed['inference']

    print("\n" + "="*30)
    print(f"results (OpenVINO INT8)")
    print(f"mean mAP50: {map50:.4f}")
    print(f"mean inference time: {avg_inference_time:.2f} ms")
    print("="*30)

if __name__ == "__main__":
    OPENVINO_DIR = "./runs/detect/runs/train/Train_scanner/weights/best_int8_openvino_model"
    DATA_YAML = "data.yaml"
    
    test_openvino_performance(OPENVINO_DIR, DATA_YAML)