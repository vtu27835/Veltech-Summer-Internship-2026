from __future__ import print_function
import argparse
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define RGBHistogram class directly
class RGBHistogram:
    def __init__(self, bins):
        # bins = [8, 8, 8] for RGB channels
        self.bins = bins

    def describe(self, image, mask=None):
        # Compute 3D RGB histogram
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

# Create ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the image dataset")
ap.add_argument("-m", "--masks", required=True, help="path to the image masks")
args = vars(ap.parse_args())

# Load image and mask paths
imagePaths = sorted(glob.glob(args["images"] + "/*.png"))
maskPaths = sorted(glob.glob(args["masks"] + "/*.png"))

data = []
target = []

# Create RGB histogram descriptor
desc = RGBHistogram([8, 8, 8])

# Loop over the dataset
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Extract features and store
    features = desc.describe(image, mask)
    data.append(features)
    target.append(imagePath.split("_")[-2])

targetNames = np.unique(target)

# Encode labels
le = LabelEncoder()
target = le.fit_transform(target)

# Split dataset
(trainData, testData, trainTarget, testTarget) = train_test_split(
    data, target, test_size=0.3, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(trainData, trainTarget)

# Print classification report
print(classification_report(testTarget, model.predict(testData), target_names=targetNames))

# Test on 10 random images
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
    imagePath = imagePaths[i]
    maskPath = maskPaths[i]

    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    features = desc.describe(image, mask)
    flower = le.inverse_transform(model.predict([features]))[0]

    print(imagePath)
    print("I think this flower is a {}".format(flower.upper()))

    cv2.imshow("Image", image)
    cv2.waitKey(0)
