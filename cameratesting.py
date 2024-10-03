import cv2


def find_connected_cameras(max_cameras=10):
    """Find all available cameras by attempting to connect to them."""
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()  # Release the camera when done testing

    return available_cameras


def show_camera_stream(camera_index):
    """Show video stream for the selected camera."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Camera {camera_index} could not be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_index}.")
            break

        # Display the frame
        cv2.imshow(f'Camera {camera_index}', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


# Find all connected cameras
available_cameras = find_connected_cameras()

if not available_cameras:
    print("No cameras found.")
else:
    print("Available cameras:")
    for index in available_cameras:
        print(f"Camera {index}")

    # Ask the user which camera to display
    selected_camera = int(input("Select a camera index to view: "))

    if selected_camera in available_cameras:
        show_camera_stream(selected_camera)
    else:
        print("Invalid camera selection.")
