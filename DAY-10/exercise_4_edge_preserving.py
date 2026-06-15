# exercise_4_edge_preserving.py

import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_noise(image, noise_level=30):
    """
    Add Gaussian noise to image
    """
    noise = np.random.normal(
        0,
        noise_level,
        image.shape
    )

    return np.clip(
        image + noise,
        0,
        255
    ).astype("uint8")


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load image!")
    exit()

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# --------------------------------------------------
# Add noise
# --------------------------------------------------

noisy = add_noise(gray, 25)

# --------------------------------------------------
# Apply different edge-preserving methods
# --------------------------------------------------

methods = {
    "Bilateral (mild)": lambda img: cv2.bilateralFilter(
        img, 9, 50, 50
    ),

    "Bilateral (medium)": lambda img: cv2.bilateralFilter(
        img, 15, 75, 75
    ),

    "Bilateral (strong)": lambda img: cv2.bilateralFilter(
        img, 21, 100, 100
    ),

    "Gaussian": lambda img: cv2.GaussianBlur(
        img, (9, 9), 0
    ),

    "Median": lambda img: cv2.medianBlur(
        img, 7
    )
}

# Bilateral filtering works best on color images
noisy_color = add_noise(image, 25)

results = {
    "Noisy": noisy
}

for name, method in methods.items():

    if "Bilateral" in name:

        result = method(noisy_color)

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2GRAY
        )

        results[name] = result

    else:
        results[name] = method(noisy)

# --------------------------------------------------
# Create comparison grid
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10)
)

plot_idx = 0

for name, img in results.items():

    row = plot_idx // 3
    col = plot_idx % 3

    axes[row, col].imshow(
        img,
        cmap="gray"
    )

    axes[row, col].set_title(name)
    axes[row, col].axis("off")

    plot_idx += 1

# Last slot = original image
axes[1, 2].imshow(
    gray,
    cmap="gray"
)

axes[1, 2].set_title(
    "Original (no noise)"
)

axes[1, 2].axis("off")

plt.tight_layout()

plt.savefig(
    "edge_preserving_comparison.png"
)

print("Saved: edge_preserving_comparison.png")

plt.show()

print("\nObservation:")
print("Bilateral filter removes noise while keeping edges sharp!")
print("Compare Bilateral vs Gaussian on the edges.")
print("Gaussian blur smooths edges, while Bilateral preserves them.")