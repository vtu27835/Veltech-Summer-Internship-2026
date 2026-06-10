# day8_color_spaces.py
# Complete guide to color spaces and channels

from __future__ import print_function
import numpy as np
import argparse
import cv2

print("=" * 70)
print("DAY 8: COLOR SPACES & CHANNELS")
print("=" * 70)

# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE
# -----------------------------------------------------------------

print("\n[Section 1] Loading image...")

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="Path to the image"
)

args = vars(ap.parse_args())

# Load image in BGR (OpenCV default)
image_bgr = cv2.imread(args["image"])

if image_bgr is None:
    print("ERROR: Could not load image!")
    exit()

height, width = image_bgr.shape[:2]

print(f"Loaded image: {width} x {height} pixels")
print("Default color space: BGR")

cv2.imshow("Original (BGR)", image_bgr)
cv2.waitKey(0)

# -----------------------------------------------------------------
# SECTION 2: COLOR SPACE CONVERSIONS
# -----------------------------------------------------------------

print("\n[Section 2] Converting between color spaces...")
print("-" * 50)

# BGR to RGB
print("\n--- BGR to RGB ---")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

cv2.imshow("BGR (OpenCV default)", image_bgr)
cv2.imshow("RGB (Human perception)", image_rgb)
cv2.waitKey(0)

print("Note: BGR and RGB look different! Red and blue channels are swapped.")

# BGR to Grayscale
print("\n--- BGR to Grayscale ---")

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", image_gray)
cv2.waitKey(0)

print(f"Grayscale shape: {image_gray.shape} (single channel)")

# BGR to HSV
print("\n--- BGR to HSV ---")

image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

cv2.imshow("HSV (Hue-Saturation-Value)", image_hsv)
cv2.waitKey(0)

print(f"HSV shape: {image_hsv.shape} (still 3 channels, but different meaning)")

# BGR to LAB
print("\n--- BGR to LAB ---")

image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

cv2.imshow("LAB (L*a*b*)", image_lab)
cv2.waitKey(0)

print("\nColor space conversions completed successfully!")