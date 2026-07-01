# Import joblib for saving and loading trained models
import joblib

# Import Linear Support Vector Classifier (SVM)
from sklearn.svm import LinearSVC

# Import custom HOG feature extractor
from pyimagesearch.hog import HOG

# Import dataset utilities
from pyimagesearch import dataset

# Import argparse
import argparse

# Create the argument parser
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d", "--dataset",
    required=True,
    help="path to the dataset file"
)
ap.add_argument(
    "-m", "--model",
    required=True,
    help="path to where the model will be stored"
)

# Parse command-line arguments
args = vars(ap.parse_args())

# Load the dataset
(digits, target) = dataset.load_digits(args["dataset"])

# Initialize the list of feature vectors
data = []

# Initialize the HOG descriptor
hog = HOG(
    orientations=18,
    pixelsPerCell=(10, 10),
    cellsPerBlock=(1, 1),
    transform=True
)

# Loop over each digit image
for image in digits:
    # Deskew the image
    image = dataset.deskew(image, 20)

    # Center the digit
    image = dataset.center_extent(image, (20, 20))

    # Extract HOG features
    hist = hog.describe(image)

    # Store the feature vector
    data.append(hist)

# Create the Linear SVM model
model = LinearSVC(random_state=42)

# Train the model
model.fit(data, target)

# Save the trained model
joblib.dump(model, args["model"])