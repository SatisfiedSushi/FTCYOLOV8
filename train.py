from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8s.pt')  # Choose the appropriate model version (e.g., yolov8s.pt, yolov8m.pt, etc.)

    # Train the model and log metrics for TensorBoard
    model.train(
        data='FTCDataSet/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # Ensure the GPU is being used
        project='runs/train',
        name='FTCYoloV8-run'
    )

if __name__ == '__main__':
    main()
