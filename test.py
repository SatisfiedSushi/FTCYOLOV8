import cv2
from ultralytics import YOLO

def main():
    # Load the trained YOLOv8 model (replace with your trained model path if different)
    model = YOLO('runs/train/FTCYolo45/weights/best.pt')  # Use the best trained model weights

    # Initialize the OpenCV video capture (0 for default camera, or specify a video file path)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection on the frame using the YOLOv8 model
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Display the frame with detections
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
