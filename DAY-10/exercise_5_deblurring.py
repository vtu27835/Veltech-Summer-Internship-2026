# exercise_5_deblurring.py

import cv2
import numpy as np
from matplotlib import pyplot as plt


def motion_blur_kernel(size, angle):
    """
    Create a motion blur kernel.
    """
    kernel = np.zeros((size, size), dtype=np.float32)

    center = size // 2
    angle_rad = np.radians(angle)

    for i in range(size):
        offset = int((i - center) * np.tan(angle_rad))
        x = center + offset

        if 0 <= x < size:
            kernel[i, x] = 1

    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)

    return kernel


def simple_deblur(image, kernel, strength=1.0):
    """
    Simple deblurring using unsharp masking.
    """

    # Apply blur with same kernel
    blurred = cv2.filter2D(image, -1, kernel)

    # Unsharp masking
    sharpened = cv2.addWeighted(
        image,
        1 + strength,
        blurred,
        -strength,
        0
    )

    return np.clip(
        sharpened,
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
# Create motion blur and attempt deblurring
# --------------------------------------------------

kernel_size = 15

for angle in [0, 45, 90, 135]:

    # Create motion blur kernel
    kernel = motion_blur_kernel(
        kernel_size,
        angle
    )

    # Apply motion blur
    blurred = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    # Attempt to deblur
    deblurred = simple_deblur(
        blurred,
        kernel,
        strength=1.5
    )

    # Create comparison image
    comparison = np.hstack([
        cv2.resize(gray, (250, 200)),
        cv2.resize(blurred, (250, 200)),
        cv2.resize(deblurred, (250, 200))
    ])

    # Add labels
    cv2.putText(
        comparison,
        f"Motion Blur: {angle}°",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "Original",
        (90, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        comparison,
        "Blurred",
        (340, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        comparison,
        "Deblurred (approx)",
        (560, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Display result
    cv2.imshow(
        f"Deblurring Challenge - {angle}° Motion",
        comparison
    )

    cv2.waitKey(0)

cv2.destroyAllWindows()

print("\nCHALLENGE COMPLETE!")
print(
    "Note: Full deblurring is complex. "
    "This uses unsharp masking as an approximation."
)