# first_image.py
# Creates a blank image and displays it

import numpy as np
import cv2

# Create a 300x300 pixel black image (3 channels for color)
# dtype="uint8" means each value is 0-255
image = np.zeros((300, 300, 3), dtype="uint8")

# Display the image
cv2.imshow("My First Image", image)

# Wait for key press (0 means wait indefinitely)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()