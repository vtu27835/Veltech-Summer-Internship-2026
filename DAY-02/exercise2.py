# exercise_2.py
import numpy as np
import cv2
# Create a 500x500 black image
black = np.zeros((500, 500, 3), dtype="uint8")
cv2.imwrite("black.jpg", black)
# Create a 500x500 white image
white = np.ones((500, 500, 3), dtype="uint8") * 255
cv2.imwrite("white.jpg", white)
# Create a 500x500 pure red image (BGR = 0, 0, 255)
red = np.zeros((500, 500, 3), dtype="uint8")
red[:, :] = (0, 0, 255)  # All pixels set to red in BGR format
cv2.imwrite("red.jpg", red)
print("Created black, white, and red images!")