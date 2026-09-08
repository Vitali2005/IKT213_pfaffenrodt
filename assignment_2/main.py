from pathlib import Path

import cv2

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


def main():
    image_path = Path(__file__).parent / "iris-1.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    padding(image, 100)
    crop(image, 200, width - 130, 200, height - 130)
    resize(image, 200, 200)
    

if __name__ == "__main__":
    main()
