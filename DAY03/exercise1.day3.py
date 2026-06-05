# exercise_1_day3.py

import cv2
import argparse

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

# Load image
image = cv2.imread(args["image"])

if image is None:
    print("Error: Could not load image.")
    exit()

# Get image dimensions
h, w = image.shape[:2]
print("Image size: {} x {}".format(w, h))

# Investigate 9 points on the image
points = [
    (0, 0, "Top-Left"),
    (w - 1, 0, "Top-Right"),
    (0, h - 1, "Bottom-Left"),
    (w - 1, h - 1, "Bottom-Right"),
    (w // 2, 0, "Top-Center"),
    (w // 2, h - 1, "Bottom-Center"),
    (0, h // 2, "Center-Left"),
    (w - 1, h // 2, "Center-Right"),
    (w // 2, h // 2, "Center")
]

for x, y, name in points:
    b, g, r = image[y, x]
    print(
        "{} ({}, {}): B={:3d}, G={:3d}, R={:3d}".format(
            name, x, y, b, g, r
        )
    )

# Display image
cv2.imshow("Image", image)
print("\nPress any key to close...")

cv2.waitKey(0)
cv2.destroyAllWindows()