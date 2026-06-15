# exercise_1_histogram_analyzer.py

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def analyze_histogram_interactive(image_path):
    """Interactive tool to explore histogram and thresholding"""

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Initial threshold
    threshold = 128

    def update(threshold_val):
        plt.clf()

        threshold_val = int(threshold_val)

        # Apply threshold
        _, thresh = cv2.threshold(
            gray,
            threshold_val,
            255,
            cv2.THRESH_BINARY
        )

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')

        # Grayscale image
        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale")
        plt.axis('off')

        # Thresholded image
        plt.subplot(2, 3, 3)
        plt.imshow(thresh, cmap='gray')
        plt.title(f"Threshold = {threshold_val}")
        plt.axis('off')

        # Histogram
        plt.subplot(2, 3, 4)

        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        plt.plot(hist, color='black')

        plt.axvline(
            x=threshold_val,
            color='red',
            linestyle='--',
            label=f'Threshold={threshold_val}'
        )

        plt.title("Grayscale Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.legend()
        plt.xlim([0, 256])

        # Cumulative histogram
        plt.subplot(2, 3, 5)

        cumsum = np.cumsum(hist)

        plt.plot(cumsum, color='blue')

        plt.axhline(
            y=cumsum[-1] * 0.5,
            color='red',
            linestyle='--',
            label='50%'
        )

        plt.title("Cumulative Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Cumulative Pixels")
        plt.legend()
        plt.xlim([0, 256])

        # Statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')

        stats = f"""
Statistics

Mean: {np.mean(gray):.1f}
Std Dev: {np.std(gray):.1f}

Min: {np.min(gray)}
Max: {np.max(gray)}

Median: {np.median(gray):.1f}

Threshold: {threshold_val}

Pixels below: {np.sum(gray < threshold_val)}
Pixels above: {np.sum(gray >= threshold_val)}
"""

        plt.text(
            0.05,
            0.5,
            stats,
            fontsize=10,
            verticalalignment='center'
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.draw()

    # Create slider
    plt.subplots_adjust(bottom=0.15)

    ax_slider = plt.axes([0.2, 0.03, 0.6, 0.03])

    slider = Slider(
        ax=ax_slider,
        label='Threshold',
        valmin=0,
        valmax=255,
        valinit=threshold,
        valstep=1
    )

    slider.on_changed(update)

    # Initial display
    update(threshold)

    plt.show()


# Run analyzer
analyze_histogram_interactive("test_image.png")