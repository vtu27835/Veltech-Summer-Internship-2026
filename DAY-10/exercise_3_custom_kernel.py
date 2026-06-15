# exercise_3_custom_kernel.py

import cv2
import numpy as np


def apply_custom_blur(image, kernel_type="sharpen", kernel_size=5):
    """
    Apply custom blur/sharpen kernels
    """

    if kernel_type == "sharpen":
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    elif kernel_type == "edge_detect":
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    elif kernel_type == "emboss":
        kernel = np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ], dtype=np.float32)

    elif kernel_type == "box_blur":
        kernel = (
            np.ones((kernel_size, kernel_size), dtype=np.float32)
            / (kernel_size ** 2)
        )

    elif kernel_type == "gaussian_custom":

        # Create Gaussian kernel
        kernel = np.zeros(
            (kernel_size, kernel_size),
            dtype=np.float32
        )

        center = kernel_size // 2
        sigma = kernel_size / 3

        for i in range(kernel_size):
            for j in range(kernel_size):

                x = i - center
                y = j - center

                kernel[i, j] = np.exp(
                    -(x * x + y * y) /
                    (2 * sigma * sigma)
                )

        kernel = kernel / np.sum(kernel)

    elif kernel_type == "motion_blur":

        kernel = np.zeros(
            (kernel_size, kernel_size),
            dtype=np.float32
        )

        kernel[kernel_size // 2, :] = (
            1.0 / kernel_size
        )

    else:
        kernel = np.ones(
            (3, 3),
            dtype=np.float32
        ) / 9

    # Apply convolution
    result = cv2.filter2D(
        image,
        -1,
        kernel
    )

    # Normalize for visualization if needed
    result = np.clip(
        result,
        0,
        255
    ).astype("uint8")

    return result, kernel


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
# Try different kernels
# --------------------------------------------------

kernels = [
    "sharpen",
    "edge_detect",
    "emboss",
    "box_blur",
    "gaussian_custom",
    "motion_blur"
]

print("Custom Kernel Effects:")
print("-" * 40)

for kernel_type in kernels:

    result, kernel = apply_custom_blur(
        gray,
        kernel_type,
        7
    )

    # Display kernel info
    print(f"\n{kernel_type.upper()}:")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel sum: {np.sum(kernel):.3f}")

    # Show filtered result
    cv2.imshow(
        f"Kernel: {kernel_type}",
        result
    )

    cv2.waitKey(0)

    # Visualize kernel
    kernel_vis = cv2.normalize(
        kernel,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    kernel_vis = cv2.resize(
        kernel_vis,
        (200, 200),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        f"Kernel: {kernel_type} (visualized)",
        kernel_vis
    )

    cv2.waitKey(0)

cv2.destroyAllWindows()