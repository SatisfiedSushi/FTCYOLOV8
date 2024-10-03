import pyheif
from PIL import Image


def heic_to_jpg(heic_file_path, jpg_file_path):
    # Read the .heic image
    heif_file = pyheif.read(heic_file_path)

    # Convert it to a format that PIL understands
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Save the image as .jpg
    image.save(jpg_file_path, "JPEG")


# Example usage
heic_to_jpg('input_image.heic', 'output_image.jpg')
