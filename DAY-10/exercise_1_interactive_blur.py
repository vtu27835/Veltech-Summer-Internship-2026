# exercise_1_interactive_blur.py

import cv2
import numpy as np


def interactive_blur_comparison(image_path):
    """
    Interactive tool to compare blurring methods
    with adjustable parameters
    """

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create trackbars
    cv2.namedWindow("Blur Comparison")

    cv2.createTrackbar(
        "Kernel Size",
        "Blur Comparison",
        5,
        31,
        lambda x: None
    )

    cv2.createTrackbar(
        "Method",
        "Blur Comparison",
        0,
        3,
        lambda x: None
    )

    cv2.createTrackbar(
        "Sigma/Bilateral",
        "Blur Comparison",
        50,
        150,
        lambda x: None
    )

    methods = [
        "Averaging",
        "Gaussian",
        "Median",
        "Bilateral"
    ]

    print("Interactive Blur Comparison Tool")
    print("-" * 40)
    print("Method: 0=Averaging, 1=Gaussian, 2=Median, 3=Bilateral")
    print("Kernel Size: 3, 5, 7, ..., 31 (odd numbers)")
    print("Sigma/Bilateral: for Gaussian sigma or bilateral sigma")
    print("Press ESC to exit")

    while True:

        # Get trackbar values
        ksize = cv2.getTrackbarPos(
            "Kernel Size",
            "Blur Comparison"
        )

        method = cv2.getTrackbarPos(
            "Method",
            "Blur Comparison"
        )

        sigma = cv2.getTrackbarPos(
            "Sigma/Bilateral",
            "Blur Comparison"
        )

        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1

        # Apply selected blur
        if method == 0:
            # Averaging
            blurred = cv2.blur(gray, (ksize, ksize))
            param_text = f"Kernel: {ksize}x{ksize}"

        elif method == 1:
            # Gaussian
            sigma_val = max(0.1, sigma / 10)

            blurred = cv2.GaussianBlur(
                gray,
                (ksize, ksize),
                sigma_val
            )

            param_text = (
                f"Kernel: {ksize}x{ksize}, "
                f"Sigma: {sigma_val:.1f}"
            )

        elif method == 2:
            # Median
            blurred = cv2.medianBlur(gray, ksize)
            param_text = f"Kernel: {ksize}x{ksize}"

        else:
            # Bilateral
            d = ksize
            sigma_color = sigma
            sigma_space = sigma

            blurred = cv2.bilateralFilter(
                image,
                d,
                sigma_color,
                sigma_space
            )

            blurred = cv2.cvtColor(
                blurred,
                cv2.COLOR_BGR2GRAY
            )

            param_text = f"d={d}, sigma={sigma}"

        # Add text to display
        display = cv2.cvtColor(
            blurred,
            cv2.COLOR_GRAY2BGR
        )

        cv2.putText(
            display,
            f"Method: {methods[method]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            param_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Show original and blurred side by side
        comparison = np.hstack([
            cv2.resize(
                cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR
                ),
                (300, 250)
            ),
            cv2.resize(
                display,
                (300, 250)
            )
        ])

        cv2.imshow(
            "Blur Comparison",
            comparison
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


# Run the interactive tool
interactive_blur_comparison("test_image.png")