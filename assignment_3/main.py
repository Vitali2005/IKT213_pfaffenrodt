from pathlib import Path
import cv2

OUTPUT_DIR = Path(__file__).parent / "solutions"


def save_image(image, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    print("Saved:", output_path)


def main():
    image_path = Path(__file__).parent / "lambo.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")


if __name__ == "__main__":
    main()
