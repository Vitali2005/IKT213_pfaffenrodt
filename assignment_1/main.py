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
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera")

    try:
        fps = camera.get(cv2.CAP_PROP_FPS)
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    finally:
        camera.release()

    output_path = Path(__file__).parent / "solutions" / "camera_outputs.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"fps: {round(fps, 2)}\n")
        file.write(f"height: {int(height)}\n")
        file.write(f"width: {int(width)}\n")

    print("Camera information written to:", output_path)


def main():
    image_path = Path(__file__).parent / "iris-1.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    print_image_information(image)

    save_camera_information()


if __name__ == "__main__":
    main()
