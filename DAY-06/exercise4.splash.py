# exercise_4_color_splash.py

import cv2
import numpy as np

# Load image
image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load test_image.png")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Create a circular mask in the center
mask = np.zeros(image.shape[:2], dtype="uint8")

center = (
    image.shape[1] // 2,
    image.shape[0] // 2
)

radius = min(
    image.shape[0],
    image.shape[1]
) // 4

cv2.circle(
    mask,
    center,
    radius,
    255,
    -1
)

# Invert mask
mask_inv = cv2.bitwise_not(mask)

# Keep color inside the circle
inside = cv2.bitwise_and(
    image,
    image,
    mask=mask
)

# Make outside grayscale
outside = cv2.bitwise_and(
    gray_color,
    gray_color,
    mask=mask_inv
)

# Combine both regions
splash = cv2.add(inside, outside)

# Draw circle boundary
cv2.circle(
    splash,
    center,
    radius,
    (0, 255, 0),
    3
)

# Display result
cv2.imshow("Color Splash Effect", splash)

cv2.waitKey(0)

# Save output
cv2.imwrite("color_splash.png", splash)

print("Color splash image saved as color_splash.png")

cv2.destroyAllWindows()