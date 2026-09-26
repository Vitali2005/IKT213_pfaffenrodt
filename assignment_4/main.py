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


def harris_corner_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray_image)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow("Harris Corner Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image(image, "harris.png")


def feature_based_image_alignment(image_to_align, reference_image, max_features, good_match_percent):
    # Using SIFT
    img1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)
    good = good[:max_features]

    MIN_MATCH_COUNT = 4
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matches_mask = mask.ravel().tolist()

        height, width = reference_image.shape[:2]

        aligned_image = cv2.warpPerspective(image_to_align, M, (width, height))

        h, w = img1.shape

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2_with_outline = cv2.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        raise ValueError(
            f"Not enough matches found: "
            f"{len(good)}/{MIN_MATCH_COUNT}"
        )

    draw_params = dict(matchColor=(0, 0, 255), singlePointColor=None, matchesMask=matches_mask, flags=2)

    matches_image = cv2.drawMatches(img1, kp1, img2_with_outline, kp2, good, None, **draw_params)

    cv2.imshow("Aligned Image", aligned_image)
    cv2.waitKey(0)

    cv2.imshow("Feature Matches", matches_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    save_image(aligned_image, "aligned.png")
    save_image(matches_image, "matches.png")


def main():
    reference_image = load_image("reference_img.png")
    image_to_align = load_image("align_this.jpg")

    harris_corner_detection(reference_image.copy())
    feature_based_image_alignment(image_to_align, reference_image, 10, 0.7)


if __name__ == "__main__":
    main()
