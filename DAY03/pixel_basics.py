# pixel_basics.py
# Learning to access and manipulate individual pixels

from __future__ import print_function
import argparse
import cv2
import numpy as np

print("=" * 50)
print("DAY 3: PIXEL BASICS")
print("=" * 50)

# ----- PART A: SETUP AND LOAD IMAGE -----
print("\n[Step 1] Loading image...")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

if image is None:
    print("ERROR: Could not load image!")
    exit()

print("Image loaded successfully!")
print("Image shape (height, width, channels):", image.shape)

# Display original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# ----- PART B: ACCESSING A SINGLE PIXEL -----
print("\n[Step 2] Accessing individual pixels...")

# Access pixel at top-left corner (0, 0)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - B:{}, G:{}, R:{}".format(b, g, r))

# Access pixel at center
height = image.shape[0]
width = image.shape[1]

center_y = height // 2
center_x = width // 2

(b, g, r) = image[center_y, center_x]

print("Pixel at center ({}, {}) - B:{}, G:{}, R:{}".format(
    center_x, center_y, b, g, r))

# Access pixel at bottom-right
(b, g, r) = image[height - 1, width - 1]

print("Pixel at bottom-right ({}, {}) - B:{}, G:{}, R:{}".format(
    width - 1, height - 1, b, g, r))