from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "solutions"


def save_image(image, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    print("Saved:", output_path)


def padding(image, border_width):
    padded = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT,
    )
    save_image(padded, "padding.png")
    return padded


def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    save_image(cropped, "crop.png")
    return cropped


def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    save_image(resized, "resize.png")
    return resized


def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]

    save_image(emptyPictureArray, "copy.png")
    return emptyPictureArray


def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray_image, "grayscale.png")
    return gray_image


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    save_image(hsv_image, "hsv.png")
    return hsv_image


def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                shifted = int(image[y, x, c]) + hue
                emptyPictureArray[y, x, c] = min(max(shifted, 0), 255)

    save_image(emptyPictureArray, "hue_shifted.png")
    return emptyPictureArray


def main():
    image_path = Path(__file__).parent / "iris-1.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    padding(image, 100)
    crop(image, 200, width - 130, 200, height - 130)
    resize(image, 200, 200)
    copy(image, emptyPictureArray)
    grayscale(image)
    hsv(image)
    hue_shifted(image, emptyPictureArray, 50)


if __name__ == "__main__":
    main()
