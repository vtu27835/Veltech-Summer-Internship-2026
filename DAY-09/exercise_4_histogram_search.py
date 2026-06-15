# exercise_4_histogram_search.py

import cv2
import numpy as np
from matplotlib import pyplot as plt


def compute_histogram_features(image):
    """Compute histogram-based features for image matching"""

    hist_features = []

    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])

        hist = cv2.normalize(hist, hist)

        hist_features.extend(hist.flatten())

    return np.array(hist_features, dtype=np.float32)


def compare_histograms(hist1, hist2):
    """Compare two histograms using correlation"""

    return cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )


# ---------------------------
# Load query image
# ---------------------------

query = cv2.imread("test_image.png")

if query is None:
    print("Error: Could not load test_image.png")
    exit()

query_hist = compute_histogram_features(query)

print("Query image histogram computed")
print(f"Feature vector size: {len(query_hist)}")

# ---------------------------
# Create image variants
# ---------------------------

variants = {
    "Original": query,
    "Brightened": cv2.add(
        query,
        np.ones(query.shape, dtype="uint8") * 50
    ),
    "Darkened": cv2.subtract(
        query,
        np.ones(query.shape, dtype="uint8") * 50
    ),
    "Flipped": cv2.flip(query, 1),
    "Rotated": cv2.rotate(
        query,
        cv2.ROTATE_90_CLOCKWISE
    ),
}

print("\nSimilarity scores (higher = more similar):")

for name, variant in variants.items():

    variant_hist = compute_histogram_features(variant)

    score = compare_histograms(
        query_hist,
        variant_hist
    )

    print(f"{name}: {score:.4f}")

# ---------------------------
# Visualization
# ---------------------------

plt.figure(figsize=(12, 6))

for idx, (name, variant) in enumerate(variants.items()):

    plt.subplot(2, 3, idx + 1)

    plt.imshow(
        cv2.cvtColor(
            variant,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(name)
    plt.axis("off")

plt.tight_layout()

plt.savefig("histogram_search_demo.png")

plt.show()