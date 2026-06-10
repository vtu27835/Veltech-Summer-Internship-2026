# exercise_1_brightness_contrast.py

import cv2
import numpy as np


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of an image.

    brightness: -255 to 255
                (negative = darker, positive = brighter)

    contrast: -127 to 127
              (negative = less contrast, positive = more contrast)
    """

    # Apply brightness
    bright = cv2.add(
        image,
        np.ones(image.shape, dtype="uint8") * brightness
    )

    # Apply contrast
    # Formula:
    # new = 127 + contrast + (old - 127) * (1 + contrast/127)
    contrast = contrast / 127.0

    adjusted = cv2.addWeighted(
        bright,
        1 + contrast,
        bright,
        0,
        -127 * contrast
    )

    return np.clip(adjusted, 0, 255).astype("uint8")


# Load image
image = cv2.imread("test_image.png")

if image is None:
    print("Error: Could not load image 'test_image.png'")
    exit()

# Create interactive adjustments
print("Press keys to adjust:")
print(" b = increase brightness")
print(" B = decrease brightness")
print(" c = increase contrast")
print(" C = decrease contrast")
print(" r = reset")
print(" ESC = exit")

bright = 0
cont = 0

while True:
    adjusted = adjust_brightness_contrast(image, bright, cont)

    # Show values on image
    cv2.putText(
        adjusted,
        f"Brightness: {bright}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        adjusted,
        f"Contrast: {cont}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Adjust Brightness/Contrast", adjusted)

    key = cv2.waitKey(100) & 0xFF

    if key == ord('b'):
        bright = min(255, bright + 10)

    elif key == ord('B'):
        bright = max(-255, bright - 10)

    elif key == ord('c'):
        cont = min(127, cont + 10)

    elif key == ord('C'):
        cont = max(-127, cont - 10)

    elif key == ord('r'):
        bright, cont = 0, 0

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()