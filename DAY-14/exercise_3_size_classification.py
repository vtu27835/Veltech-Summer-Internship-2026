# exercise_3_size_classification.py

import cv2
import numpy as np


def classify_by_size(image_path):
    """Classify objects by size (small, medium, large)"""

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours_info = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[-2]

    # Calculate areas
    areas = [
        cv2.contourArea(c)
        for c in contours
        if cv2.contourArea(c) > 50
    ]

    if not areas:
        print("No objects found")
        return

    # Determine thresholds
    max_area = max(areas)

    thresholds = {
        'small': max_area * 0.2,
        'medium': max_area * 0.5,
        'large': max_area * 0.8
    }

    # Classify
    size_counts = {
        'small': 0,
        'medium': 0,
        'large': 0
    }

    result = image.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 50:
            continue

        if area < thresholds['small']:
            size = 'small'
        elif area < thresholds['medium']:
            size = 'medium'
        else:
            size = 'large'

        size_counts[size] += 1

        # Color code
        colors = {
            'small': (255, 0, 0),    # Blue
            'medium': (0, 255, 0),   # Green
            'large': (0, 0, 255)     # Red
        }

        cv2.drawContours(result, [contour], -1, colors[size], 2)

        # Label
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                result,
                size,
                (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

    # Print results
    print("\nSize Classification Results")
    print("-" * 40)

    for size, count in size_counts.items():
        print(f"{size}: {count} objects")

    cv2.imshow("Size Classification", result)
    cv2.imwrite("size_classification.png", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run
classify_by_size("test_image.png")