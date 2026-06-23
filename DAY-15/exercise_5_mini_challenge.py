# exercise_5_mini_challenge.py

import cv2
import numpy as np


def detect_and_analyze_faces(image_path, cascade_path):
    """
    Complete face detection pipeline:
    1. Detect faces
    2. Draw bounding boxes
    3. Extract and save each face
    4. Analyze face properties (size, position)
    """

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image")
        return

    height, width = image.shape[:2]

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        print("Could not load cascade")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    print("=" * 50)
    print("FACE DETECTION ANALYSIS")
    print("=" * 50)
    print(f"Image size: {width} x {height}")
    print(f"Faces detected: {len(faces)}")
    print()

    # Analyze each face
    result = image.copy()

    for i, (x, y, w, h) in enumerate(faces):

        # Draw bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Label face
        cv2.putText(
            result,
            f"Face {i + 1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Extract face
        face_roi = image[y:y + h, x:x + w]
        cv2.imwrite(f"challenge_face_{i + 1}.png", face_roi)

        # Analyze
        center_x = x + w // 2
        center_y = y + h // 2

        image_center_x = width // 2
        image_center_y = height // 2

        distance = np.sqrt(
            (center_x - image_center_x) ** 2 +
            (center_y - image_center_y) ** 2
        )

        print(f"Face {i + 1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w} x {h} pixels")
        print(f"  Center: ({center_x}, {center_y})")
        print(f"  Distance from image center: {distance:.0f} pixels")
        print(f"  Face ratio: {w / h:.2f}")
        print(f"  Saved as: challenge_face_{i + 1}.png")
        print()

    # Add summary
    cv2.putText(
        result,
        f"Total Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection Challenge", result)
    cv2.waitKey(0)

    cv2.imwrite("challenge_result.png", result)

    return faces


# Run the challenge
detect_and_analyze_faces(
    "images/family.jpg",
    "cascades/haarcascade_frontalface_default.xml"
)

cv2.destroyAllWindows()