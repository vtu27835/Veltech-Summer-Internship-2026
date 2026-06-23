# exercise_3_face_roi.py

import cv2


def detect_faces_with_roi(image_path, cascade_path, roi=None):
    """Detect faces in a specific region of interest"""

    image = cv2.imread(image_path)
    if image is None:
        return []

    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If ROI is specified, crop the image
    if roi is not None:
        x, y, w, h = roi

        roi_image = gray[y:y+h, x:x+w]

        faces = cascade.detectMultiScale(
            roi_image,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # Adjust coordinates back to original image
        faces = [(fx + x, fy + y, fw, fh)
                 for (fx, fy, fw, fh) in faces]

    else:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

    # Draw results
    result = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Show ROI if specified
    if roi is not None:
        cv2.rectangle(
            result,
            (roi[0], roi[1]),
            (roi[0] + roi[2], roi[1] + roi[3]),
            (255, 0, 0),
            2
        )

        cv2.putText(
            result,
            "ROI",
            (roi[0], roi[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    cv2.imshow("Face Detection with ROI", result)
    cv2.waitKey(0)

    return faces


# Detect faces only in the center region
faces = detect_faces_with_roi(
    "images/family.jpg",
    "cascades/haarcascade_frontalface_default.xml",
    roi=(100, 100, 300, 300)
)

print(f"Found {len(faces)} faces in ROI")

cv2.destroyAllWindows()