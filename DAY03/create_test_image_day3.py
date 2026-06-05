import numpy as np
import cv2

# Create a gradient image to see pixel values clearly
image = np.zeros((300, 400, 3), dtype="uint8")

# Fill with gradient
for y in range(300):
    for x in range(400):

        # Different colors for different regions
        if x < 133:
            image[y, x] = (x % 256, 0, 0)      # Blue gradient
        elif x < 266:
            image[y, x] = (0, x % 256, 0)      # Green gradient
        else:
            image[y, x] = (0, 0, x % 256)      # Red gradient

cv2.imwrite("test_gradient.png", image)
print("Created test_gradient.png")

# Create a simple pattern image
pattern = np.zeros((300, 400, 3), dtype="uint8")

# Blue square
cv2.rectangle(pattern, (50, 50), (150, 150), (255, 0, 0), -1)

# Green square
cv2.rectangle(pattern, (200, 50), (300, 150), (0, 255, 0), -1)

# Red square
cv2.rectangle(pattern, (50, 200), (150, 300), (0, 0, 255), -1)

cv2.imwrite("test_pattern.png", pattern)
print("Created test_pattern.png")