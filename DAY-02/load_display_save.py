from __future__ import print_function

import argparse
import cv2

# ----- PART A: SETUP COMMAND-LINE ARGUMENTS -----

print("Step 1: Setting up command-line arguments...")

ap = argparse.ArgumentParser()

ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="Path to the image file"
)

args = vars(ap.parse_args())

print("Image path provided:", args["image"])
print("")

# ----- PART B: LOAD THE IMAGE -----

print("Step 2: Loading image from disk...")

image = cv2.imread(args["image"])

if image is None:
    print("ERROR: Could not load image. Check the file path.")
    exit()

print("Image loaded successfully!")
print("")

# ----- PART C: EXAMINE IMAGE PROPERTIES -----

print("Step 3: Examining image properties...")

height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]

print("Image dimensions:")
print(" - Width: {} pixels".format(width))
print(" - Height: {} pixels".format(height))
print(" - Channels: {} (BGR color)".format(channels))
print(" - Total pixels: {}".format(width * height))
print("")

# ----- PART D: DISPLAY THE IMAGE -----

print("Step 4: Displaying image on screen...")
print(" (Press any key on the image window to continue)")

cv2.imshow("Original Image", image)
cv2.waitKey(0)

print("Key pressed! Continuing...")
print("")

# ----- PART E: SAVE THE IMAGE -----

print("Step 5: Saving image in new format...")

success = cv2.imwrite("saved_image.jpg", image)

if success:
    print("SUCCESS: Image saved as 'saved_image.jpg'")
else:
    print("FAILED: Could not save image")

success_png = cv2.imwrite("saved_image.png", image)

if success_png:
    print("SUCCESS: Image saved as 'saved_image.png'")

print("")
print("=" * 40)
print("Script completed successfully!")
print("Check your folder for the saved images.")
print("=" * 40)

cv2.destroyAllWindows()