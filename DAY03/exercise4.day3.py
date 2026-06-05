# exercise_4_day3.py

import numpy as np
import cv2

# Create an image that shows pixel value mapping
value_map = np.zeros((256, 256, 3), dtype="uint8")

# Fill with pixel values from 0-255
for y in range(256):
    for x in range(256):
        value_map[y, x] = (y, y, y)  # All channels same = grayscale

# Display the image
cv2.imshow("Pixel Value Map (0=black, 255=white)", value_map)
cv2.waitKey(0)

# Show different intensities
print("Value at (0,0):", value_map[0, 0])          # [0 0 0]
print("Value at (255,255):", value_map[255, 255])  # [255 255 255]

cv2.destroyAllWindows()