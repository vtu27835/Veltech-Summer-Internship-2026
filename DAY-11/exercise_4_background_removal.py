import cv2
import numpy as np

def remove_background(image, threshold_method="otsu"):
    """
    Remove background from image using thresholding
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold_method == "otsu":
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    elif threshold_method == "adaptive":
        mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    else:
        _, mask = cv2.threshold(
            blurred,
            127,
            255,
            cv2.THRESH_BINARY
        )

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to original
    result = cv2.bitwise_and(image, image, mask=mask)

    # Create transparent background (RGBA)
    if result.shape[2] == 3:
        b, g, r = cv2.split(result)
        rgba = cv2.merge([b, g, r, mask])
        result = rgba

    return result, mask


# Load test image
image = cv2.imread("test_image.png")

if image is not None:

    # Resize for faster processing
    image = cv2.resize(image, (400, 300))

    # Remove background
    result, mask = remove_background(image)

    # Display results
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Background Removed", result[:, :, :3])

    # Create composite with colored background
    colored_bg = np.full_like(image, (255, 100, 100))

    composite = np.where(
        mask[:, :, np.newaxis] == 255,
        image,
        colored_bg
    )

    cv2.imshow("Composite (Red Background)", composite)

    cv2.waitKey(0)

    cv2.imwrite("background_removed.png", result)
    print("Saved: background_removed.png")

else:
    print("Error: Could not load test_image.png")

cv2.destroyAllWindows()