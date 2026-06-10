# exercise_3_color_isolation.py

import cv2
import numpy as np

# Load image
image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load test_image.png")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV ranges for colors
color_ranges = {
    "Red": [
        ([0, 50, 50], [10, 255, 255]),
        ([160, 50, 50], [179, 255, 255])
    ],

    "Green": [
        ([40, 50, 50], [80, 255, 255])
    ],

    "Blue": [
        ([100, 50, 50], [130, 255, 255])
    ],

    "Yellow": [
        ([20, 50, 50], [30, 255, 255])
    ],

    "Purple": [
        ([130, 50, 50], [160, 255, 255])
    ]
}

# Process each color
for color_name, ranges in color_ranges.items():

    # Create mask
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for lower, upper in ranges:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        current_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, current_mask)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Keep selected color, make others grayscale
    result = np.where(
        mask[:, :, np.newaxis] > 0,
        image,
        gray_3ch
    )

    # Display result
    cv2.imshow(f"Keep Only {color_name}", result)

    # Save image
    filename = f"color_isolation_{color_name.lower()}.png"
    cv2.imwrite(filename, result)

    print(f"Saved: {filename}")

    cv2.waitKey(0)

cv2.destroyAllWindows()