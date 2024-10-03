import tensorflow as tf
import numpy as np
import cv2

# Define paths
tflite_model_path = 'FTCYolo45.tflite'  # Path to your TFLite model

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)

# Define expected input shape
input_shape = input_details[0]['shape']  # Should be [1, 3, 640, 640] (NCHW format)
print(f"Expected input shape: {input_shape}")

# Define class names from the YAML file
class_names = ['blue-specimen', 'red-specimen', 'yellow-specimen']

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Define a function to extract and draw bounding boxes from the model output
def draw_boxes(image, boxes, scores, class_ids, class_names, conf_threshold=0.5):
    """Draw bounding boxes and labels on an image."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = box

        # Draw the rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw the label
        label = f"{class_names[int(class_id)]}: {score:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_output(output_data, frame_shape, threshold=0.5):
    """Process the model output to extract bounding boxes, confidence scores, and class IDs."""
    boxes = []
    scores = []
    class_ids = []

    # The output shape is (1, 7, 8400) - we need to iterate over each detection
    for detection in output_data[0]:
        x_center, y_center, width, height, conf, class_score, class_id = detection

        # Filter out low confidence scores
        if conf * class_score < threshold:
            continue

        # Convert to corner coordinates
        x_min = int((x_center - width / 2) * frame_shape[1])
        y_min = int((y_center - height / 2) * frame_shape[0])
        x_max = int((x_center + width / 2) * frame_shape[1])
        y_max = int((y_center + height / 2) * frame_shape[0])

        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(conf * class_score)
        class_ids.append(class_id)

    return boxes, scores, class_ids

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize and preprocess the frame to match the input shape of the model
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))  # Resize to (640, 640)

    # Convert BGR to RGB (TFLite models usually require RGB format)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Transpose the frame to match the expected input shape of [1, 3, 640, 640] (NCHW format)
    frame_transposed = np.transpose(frame_rgb, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)

    # Add batch dimension and convert to float32
    input_data = np.expand_dims(frame_transposed, axis=0).astype(np.float32)  # Shape: (1, 3, 640, 640)

    # Verify the input shape matches the model's expectations
    print(f"Input data shape: {input_data.shape}")

    # Set the tensor for the TFLite model
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the model's output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output data shape: {output_data.shape}")

    # Process output to extract bounding boxes, scores, and class IDs
    boxes, scores, class_ids = process_output(output_data, frame.shape)

    print(f"Number of detections: {len(boxes)}")

    # Draw bounding boxes on the frame if there are any detections
    if len(boxes) > 0:
        draw_boxes(frame, boxes, scores, class_ids, class_names)

    # Display the frame
    cv2.imshow('TFLite Model Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
