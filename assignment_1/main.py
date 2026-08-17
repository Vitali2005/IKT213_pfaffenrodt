from pathlib import Path
import cv2


def print_image_information(image):
    height, width, channels = image.shape
    print("Image height:", height)
    print("Image width:", width)
    print("Image channels:", channels)
    print("Image size:", image.size)
    print("Image data type:", image.dtype)


def save_camera_information():
    ...


def main():
    image_path = Path(__file__).parent / "iris-1.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    print_image_information(image)


if __name__ == "__main__":
    main()
