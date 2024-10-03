import os
import cv2


# Function to convert video to frames and save them in the output directory
def video_to_images(video_path, output_dir, frame_rate=1):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get the total number of frames
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break  # If no frame is returned, exit the loop

        # Save every nth frame (based on frame_rate)
        if frame_count % frame_rate == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)  # Save the frame as an image
            print(f"Saved: {frame_name}")
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {saved_frame_count} frames from {video_path}.")


# Function to convert all videos in a directory
def convert_videos_to_images(video_dir, output_dir, frame_rate=1):
    # Iterate over all files in the video directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith((".mp4", ".avi", ".MOV", ".mkv")):  # Add other video formats as needed
            video_path = os.path.join(video_dir, video_file)
            # Create a subdirectory for each video to store its frames
            video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
            video_to_images(video_path, video_output_dir, frame_rate)


# Set the paths for the video directory and the output directory
video_directory = "Other"  # Replace with the path to your video directory
output_directory = "OtherConverted"  # Replace with the path to your output directory

# Set the frame rate for extracting images (e.g., 1 frame per second)
frame_rate = 1  # Extract 1 frame every second

# Convert all videos in the video directory to images
convert_videos_to_images(video_directory, output_directory, frame_rate)
