# exercise_3.py
import argparse
import cv2
# Create argument parser
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="Path to image"
)
args = vars(ap.parse_args())
# Load image
image = cv2.imread(args["image"])
if image is None:
    print("Error: Could not load image")
else:
    h, w, c = image.shape
    print("File:", args["image"])
    print("Dimensions: {} x {}".format(w, h))
    print("Total pixels:", w * h)
    print("Channels:", c)
    print("Memory size:", image.nbytes, "bytes")
    print("Memory size:", image.nbytes / 1024, "KB")