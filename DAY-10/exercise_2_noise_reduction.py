# exercise_2_noise_reduction.py

import cv2
import numpy as np
from matplotlib import pyplot as plt


def noise_reduction_pipeline(image, noise_type="auto"):
    """
    Automatically choose the best blurring method
    based on noise type.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect noise type (simplified)
    if noise_type == "saltpepper":
        result = cv2.medianBlur(gray, 5)
        method = "Median Blur (best for salt-and-pepper)"

    elif noise_type == "gaussian":
        result = cv2.GaussianBlur(gray, (5, 5), 0)
        method = "Gaussian Blur (best for Gaussian noise)"

    elif noise_type == "auto":

        # Check if likely salt-and-pepper noise
        unique, counts = np.unique(gray, return_counts=True)

        extreme_pixels = 0

        for val, count in zip(unique, counts):
            if val <= 5 or val >= 250:
                extreme_pixels += count

        if extreme_pixels / gray.size > 0.05:  # >5% extreme values
            result = cv2.medianBlur(gray, 5)
            method = "Median Blur (auto-detected)"
        else:
            result = cv2.GaussianBlur(gray, (5, 5), 0)
            method = "Gaussian Blur (auto-detected)"

    else:
        result = cv2.blur(gray, (5, 5))
        method = "Averaging Blur"

    return result, method


# -------------------------------------------------------
# Load image
# -------------------------------------------------------

image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load image!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# -------------------------------------------------------
# Noise generation functions
# -------------------------------------------------------

def add_salt_pepper(img, prob=0.02):
    output = img.copy()

    # Salt
    salt_mask = np.random.random(img.shape) < (prob / 2)
    output[salt_mask] = 255

    # Pepper
    pepper_mask = np.random.random(img.shape) < (prob / 2)
    output[pepper_mask] = 0

    return output


def add_gaussian(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)

    return np.clip(
        img + noise,
        0,
        255
    ).astype("uint8")


# -------------------------------------------------------
# Create test images
# -------------------------------------------------------

test_images = {
    "Original": gray,
    "Salt & Pepper": add_salt_pepper(gray, 0.03),
    "Gaussian": add_gaussian(gray, 20),
    "Mixed": add_gaussian(
        add_salt_pepper(gray, 0.02),
        15
    )
}

# -------------------------------------------------------
# Apply pipeline and visualize results
# -------------------------------------------------------

fig, axes = plt.subplots(4, 3, figsize=(15, 20))

for idx, (name, noisy) in enumerate(test_images.items()):

    # Column 1: Noisy Image
    axes[idx, 0].imshow(noisy, cmap="gray")
    axes[idx, 0].set_title(f"{name} (Noisy)")
    axes[idx, 0].axis("off")

    # Apply pipeline with different assumptions
    result_sp, method_sp = noise_reduction_pipeline(
        cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR),
        "saltpepper"
    )

    result_gauss, method_gauss = noise_reduction_pipeline(
        cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR),
        "gaussian"
    )

    result_auto, method_auto = noise_reduction_pipeline(
        cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR),
        "auto"
    )

    # Column 2: Salt-Pepper Assumption
    axes[idx, 1].imshow(result_sp, cmap="gray")
    axes[idx, 1].set_title(f"Salt-Pepper: {method_sp[:20]}")
    axes[idx, 1].axis("off")

    # Column 3: Auto Detection
    axes[idx, 2].imshow(result_auto, cmap="gray")
    axes[idx, 2].set_title(f"Auto: {method_auto[:20]}")
    axes[idx, 2].axis("off")

plt.tight_layout()

plt.savefig("noise_reduction_pipeline.png")
print("Saved: noise_reduction_pipeline.png")

plt.show()