# exercise_1_custom_pipeline.py

import cv2
import imutils
import numpy as np

def my_custom_pipeline(image, effect="vintage"):
    """Create your own custom image processing pipeline"""

    result = image.copy()

    if effect == "vintage":
        # Sepia/warm tone effect
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = cv2.addWeighted(result, 0.7, gray, 0.3, 30)

        # Add vignette
        h, w = result.shape[:2]
        mask = np.zeros((h, w), dtype="uint8")

        cv2.ellipse(
            mask,
            (w // 2, h // 2),
            (w // 2, h // 2),
            0,
            0,
            360,
            255,
            -1
        )

        result = cv2.bitwise_and(result, result, mask=mask)

    elif effect == "cyberpunk":
        # Increase contrast and add blue/purple tint
        result = cv2.addWeighted(result, 1.5, result, 0, -50)

        # Boost blue channel
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.5, 0, 255)

    elif effect == "sketch":
        # Convert to pencil sketch
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    return result


# Test your pipeline
image = cv2.imread("challenge_main.png")

for effect in ["vintage", "cyberpunk", "sketch"]:
    processed = my_custom_pipeline(image, effect)

    cv2.imshow(f"Effect: {effect}", processed)
    cv2.waitKey(0)

    cv2.imwrite(f"custom_{effect}.png", processed)

cv2.destroyAllWindows()