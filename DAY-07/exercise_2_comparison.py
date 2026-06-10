# exercise_2_comparison.py

import cv2
import imutils
import numpy as np

def enhance_image(image):
    """Apply a series of enhancements"""

    result = image.copy()

    # 1. Increase brightness slightly
    result = cv2.add(
        result,
        np.ones(result.shape, dtype="uint8") * 20
    )

    # 2. Increase contrast
    result = cv2.addWeighted(result, 1.3, result, 0, -40)

    # 3. Sharpen (using custom kernel)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    result = cv2.filter2D(result, -1, kernel)

    return result


image = cv2.imread("challenge_main.png")

enhanced = enhance_image(image)

# Create before/after comparison
comparison = np.hstack([image, enhanced])

cv2.putText(
    comparison,
    "BEFORE",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.putText(
    comparison,
    "AFTER",
    (image.shape[1] + 10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Before vs After Enhancement", comparison)
cv2.waitKey(0)

cv2.imwrite("before_after.png", comparison)

cv2.destroyAllWindows()