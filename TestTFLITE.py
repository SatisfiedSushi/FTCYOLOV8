import tensorflow as tf
import numpy as np
import cv2


def validate_input_shape(interpreter, input_shape):
    """
    Validate the input shape of the TensorFlow Lite model interpreter.
    :param interpreter: TFLite interpreter
    :param input_shape: Expected input shape
    :return: Boolean indicating whether the shape is valid
    """
    input_details = interpreter.get_input_details()
    model_input_shape = input_details[0]['shape']
    print(f"Model input shape: {model_input_shape}, Expected input shape: {input_shape}")

    if tuple(model_input_shape) == tuple(input_shape):
        print("Input shape is valid.")
        return True
    else:
        print("Input shape mismatch! Model may not work as expected.")
        return False


def draw_bounding_boxes(frame, boxes, scores, classes, threshold=0.5):
    """
    Draw bounding boxes and labels on the frame.
    :param frame: The original image frame
    :param boxes: Bounding box coordinates (normalized [ymin, xmin, ymax, xmax])
    :param scores: Confidence scores for each detected object
    :param classes: Detected class labels
    :param threshold: Minimum score threshold for displaying a bounding box
    """
    height, width, _ = frame.shape

    for i in range(len(boxes)):
        if scores[i] >= threshold:
            # Get bounding box coordinates and scale to original frame size
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)

            # Draw rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw label and confidence
            label = f"Class {int(classes[i])}: {scores[i]:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main(tflite_model_path='FTCYolo45.tflite', input_shape=(1, 3, 640, 640)):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Validate input shape (you can remove this check if not needed)
    if not validate_input_shape(interpreter, input_shape):
        print("Warning: Input shape mismatch! Adjusting input shape...")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print the output details to see available outputs and indices
    print("Output Details:", output_details)

    # Capture video using OpenCV (You can use your webcam or any video file)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the frame to match input shape (NHWC format)
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[3]))  # Resize to (640, 640)

        # Change the order of the dimensions to match the TFLite model (NHWC -> NCHW)
        frame_resized = np.transpose(frame_resized, (2, 0, 1))  # Convert NHWC to NCHW

        # Add batch dimension
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32)  # Shape: (1, 3, 640, 640)

        # Set the tensor for the TFLite model
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)

        # Run the interpreter
        interpreter.invoke()

        # Get the single output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: (7, 8400)
        print("Output data shape:", output_data)

        # Split the tensor to extract bounding boxes, scores, and class labels
        boxes = output_data[0:4, :].T  # Transpose to get (8400, 4)
        scores = output_data[4, :]  # Shape: (8400,)
        classes = output_data[5, :]  # Shape: (8400,)
        num_detections = int(output_data[6, 0])
        print("Number of detections:", num_detections)

        # Draw bounding boxes on the frame
        draw_bounding_boxes(frame, boxes, scores, classes)

        # Display the frame with bounding boxes
        cv2.imshow('TFLite Model Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
