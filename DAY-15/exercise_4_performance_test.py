# exercise_4_performance_test.py

import cv2
import time
import numpy as np


def performance_test(image_path, cascade_path):
    """Test face detection performance with different parameters"""

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        print(f"Could not load cascade: {cascade_path}")
        return

    parameter_sets = [
        (1.05, 3, "Slow, accurate"),
        (1.1, 5, "Balanced"),
        (1.2, 5, "Fast, may miss faces"),
        (1.3, 5, "Very fast"),
        (1.1, 10, "Accurate, fewer false positives"),
        (1.05, 10, "Most accurate, slowest")
    ]

    results = []

    for scale, neighbors, desc in parameter_sets:

        start_time = time.time()

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=neighbors
        )

        end_time = time.time()

        elapsed = (end_time - start_time) * 1000  # milliseconds

        result = image.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        cv2.putText(
            result,
            f"{len(faces)} faces",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            desc,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        cv2.putText(
            result,
            f"{elapsed:.1f}ms",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        results.append(result)

        print(f"{desc}: {len(faces)} faces in {elapsed:.1f}ms")

    # Display first 4 results side-by-side
    comparison = np.hstack(results[:4])

    cv2.imshow("Performance Comparison", comparison)
    cv2.waitKey(0)

    cv2.imwrite("performance_comparison.png", comparison)


# Run test
performance_test(
    "images/obama.png",
    "cascades/haarcascade_frontalface_default.xml"
)

cv2.destroyAllWindows()