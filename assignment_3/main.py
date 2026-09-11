from pathlib import Path
import cv2

OUTPUT_DIR = Path(__file__).parent / "solutions"


def save_image(image, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    print("Saved:", output_path)


def sobel_edge_detection(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)

    # Sobel edges with x and y derivatives and magnitude
    sobel_x = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    sobel_y = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Sobel edges with mixed derivative
    sobel_mixed_derivative = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1)
    sobel_edges_mixed_derivative = cv2.normalize(sobel_mixed_derivative, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow("Sobel X", sobel_x)
    cv2.waitKey( 0)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.waitKey(0)
    cv2.imshow('Sobel gradient magnitude', sobel_edges_magnitude)
    cv2.waitKey(0)
    cv2.imshow('Sobel mixed derivative', sobel_edges_mixed_derivative)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    save_image(sobel_edges_magnitude, "sobel_edges_magnitude.png")
    save_image(sobel_edges_mixed_derivative, "sobel_edges_mixed_derivative.png")


def canny_edge_detection(image, threshold_1, threshold_2):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)

    canny_edges = cv2.Canny(image=blurred_image, threshold1=threshold_1, threshold2=threshold_2)

    cv2.imshow("Canny Edge Detection", canny_edges)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    save_image(canny_edges, "canny_edges.png")


def main():
    image_path = Path(__file__).parent / "lambo.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    sobel_edge_detection(image)
    canny_edge_detection(image, 50, 50)


if __name__ == "__main__":
    main()
