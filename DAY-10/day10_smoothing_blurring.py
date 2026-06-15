# day10_smoothing_blurring.py
# Complete guide to image smoothing and blurring

from __future__ import print_function
import numpy as np
import argparse
import cv2

print("=" * 70)
print("DAY 10: SMOOTHING AND BLURRING")
print("=" * 70)

# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE AND CREATE NOISY VERSIONS
# -----------------------------------------------------------------

print("\n[Section 1] Loading image and creating noisy versions...")

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="Path to the image"
)

args = vars(ap.parse_args())

# Load image
image = cv2.imread(args["image"])

if image is None:
    print("ERROR: Could not load image!")
    exit()

height, width = image.shape[:2]
print(f"Loaded: {width} x {height} pixels")

# Convert to grayscale for simpler demonstration
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def add_salt_pepper_noise(img, prob=0.02):
    """
    Add salt-and-pepper noise to image
    """
    output = img.copy()

    num_salt = np.ceil(prob * img.size * 0.5)
    num_pepper = np.ceil(prob * img.size * 0.5)

    # Add salt (white)
    coords = [
        np.random.randint(0, i, int(num_salt))
        for i in img.shape
    ]
    output[coords[0], coords[1]] = 255

    # Add pepper (black)
    coords = [
        np.random.randint(0, i, int(num_pepper))
        for i in img.shape
    ]
    output[coords[0], coords[1]] = 0

    return output


def add_gaussian_noise(img, mean=0, sigma=25):
    """
    Add Gaussian noise to image
    """
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise

    return np.clip(noisy, 0, 255).astype("uint8")


# Create noisy versions
noisy_saltpepper = add_salt_pepper_noise(gray, 0.03)
noisy_gaussian = add_gaussian_noise(gray, 0, 20)

# Display images
cv2.imshow("Original (Grayscale)", gray)
cv2.imshow("Salt-and-Pepper Noise", noisy_saltpepper)
cv2.imshow("Gaussian Noise", noisy_gaussian)

cv2.waitKey(0)

print("Created noisy images for testing blurring methods.")