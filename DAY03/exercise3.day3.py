# exercise_3_day3.py

import numpy as np
import cv2

# Create a smooth color gradient
gradient = np.zeros((300, 512, 3), dtype="uint8")

# Red channel gradient (left to right)
for x in range(512):
    gradient[:, x, 2] = x * 255 // 511

# Green channel gradient (top to bottom)
for y in range(300):
    gradient[y, :, 1] = y * 255 // 299

# Blue channel gradient (diagonal)
for y in range(300):
    for x in range(512):
        gradient[y, x, 0] = (x + y) * 255 // (511 + 299)

# Display the gradient image
cv2.imshow("Color Gradient", gradient)
cv2.waitKey(0)

# Save the image
cv2.imwrite("gradient.png", gradient)

cv2.destroyAllWindows()

print("Saved gradient.png")