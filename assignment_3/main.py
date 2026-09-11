from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "solutions"


def load_image(image_name):
    image_path = Path(__file__).parent / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def save_image(image, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    print("Saved:", output_path)


def sobel_edge_detection(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)

    # Sobel using separate x/y derivatives and gradient magnitude
    sobel_x = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    sobel_y = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow("Sobel X", sobel_x)
    cv2.waitKey(0)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.waitKey(0)
    cv2.imshow('Sobel gradient magnitude', sobel_edges_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image(sobel_edges_magnitude, "sobel_edges_magnitude.png")

    # Sobel using the mixed x/y derivative
    sobel_mixed_derivative = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1)
    sobel_edges_mixed_derivative = cv2.normalize(sobel_mixed_derivative, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imshow('Sobel mixed derivative', sobel_edges_mixed_derivative)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image(sobel_edges_mixed_derivative, "sobel_edges_mixed_derivative.png")


def canny_edge_detection(image, threshold_1, threshold_2):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)

    canny_edges = cv2.Canny(image=blurred_image, threshold1=threshold_1, threshold2=threshold_2)

    cv2.imshow("Canny Edge Detection", canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image(canny_edges, "canny_edges.png")


def template_match(image, template):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = grayscale_template.shape[::-1]
    result = cv2.matchTemplate(grayscale_image, grayscale_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    location = np.where(result >= threshold)

    for pt in zip(*location[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imshow("Template Matching", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image(image, "template_matching.png")


def resize(image, scale_factor:int, up_or_down: str):
    if up_or_down == "up":
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        resized = cv2.pyrUp(image, dstsize=(new_width, new_height))

    elif up_or_down == "down":
        new_width = int(image.shape[1] / scale_factor)
        new_height = int(image.shape[0] / scale_factor)
        resized = cv2.pyrDown(image, dstsize=(new_width, new_height))

    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image(resized, f"resized_{up_or_down}.png")


def main():
    lambo_image = load_image("lambo.png")
    shapes_image = load_image("shapes-1.png")
    template = load_image("shapes_template.jpg")

    sobel_edge_detection(lambo_image)
    canny_edge_detection(lambo_image, 50, 50)
    template_match(shapes_image, template)
    resize(lambo_image, 2, "up")
    resize(lambo_image, 2, "down")


if __name__ == "__main__":
    main()
