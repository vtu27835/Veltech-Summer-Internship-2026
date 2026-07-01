from __future__ import print_function

import joblib
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import argparse
import mahotas
import cv2

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model",
    required=True,
    help="path to where the model will be stored"
)
ap.add_argument(
    "-i", "--image",
    required=True,
    help="path to the image file"
)
args = vars(ap.parse_args())

# Load the trained model
model = joblib.load(args["model"])

# Initialize the HOG descriptor
hog = HOG(
    orientations=18,
    pixelsPerCell=(10, 10),
    cellsPerBlock=(1, 1),
    transform=True
)

# Load the image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur and detect edges
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

# Find contours (works with OpenCV 4)
cnts, _ = cv2.findContours(
    edged.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Sort contours from left to right
cnts = sorted(
    [(c, cv2.boundingRect(c)[0]) for c in cnts],
    key=lambda x: x[1]
)

# Loop over the contours
for (c, _) in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 7 and h >= 20:
        # Extract ROI
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()

        # Otsu threshold
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)

        # Deskew and center
        thresh = dataset.deskew(thresh, 20)
        thresh = dataset.center_extent(thresh, (20, 20))

        # Display processed digit
        cv2.imshow("thresh", thresh)

        # Extract HOG features
        hist = hog.describe(thresh)

        # Predict the digit
        digit = model.predict([hist])[0]

        print("I think that number is: {}".format(digit))

        # Draw prediction
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.putText(
            image,
            str(digit),
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

# Show final image
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()