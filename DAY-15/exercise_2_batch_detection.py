# exercise_2_batch_detection.py

import cv2
import os
import glob


def batch_face_detection(image_dir, cascade_path, output_dir="detected_faces"):
    """Detect faces in all images in a directory"""

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load cascade
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("Could not load cascade")
        return

    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    print(f"Found {len(image_files)} images")

    for image_path in image_files:
        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Save result
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{filename}")

        cv2.imwrite(output_path, image)

        print(f"{filename}: {len(faces)} faces detected")


# Run batch detection
batch_face_detection(
    "images/",
    "cascades/haarcascade_frontalface_default.xml"
)

cv2.destroyAllWindows()