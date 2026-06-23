# exercise_1_cascade_comparison.py

import cv2
import os
import glob
import numpy as np


def test_all_cascades(image_path, cascade_dir):
    """Test all available cascade files on an image"""

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
        return

    # Find all cascade files
    cascade_files = glob.glob(os.path.join(cascade_dir, "*.xml"))

    results = []

    for cascade_file in cascade_files:
        name = os.path.basename(cascade_file)
        cascade = cv2.CascadeClassifier(cascade_file)

        if cascade.empty():
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        result = image.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            result,
            f"{name[:20]}: {len(faces)} faces",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        results.append(result)
        print(f"{name}: found {len(faces)} faces")

    # Display all results
    if results:
        # Create a grid
        cols = min(4, len(results))
        rows = (len(results) + cols - 1) // cols

        grid_height = 200 * rows
        grid_width = 300 * cols

        grid = np.zeros((grid_height, grid_width, 3), dtype="uint8")

        for i, img in enumerate(results[:cols * rows]):
            row = i // cols
            col = i % cols

            img_resized = cv2.resize(img, (300, 200))

            grid[
                row * 200:(row + 1) * 200,
                col * 300:(col + 1) * 300
            ] = img_resized

        cv2.imshow("All Cascades Comparison", grid)
        cv2.waitKey(0)
        cv2.imwrite("cascade_comparison.png", grid)


# Run the test
test_all_cascades("images/obama.png", "cascades/")
cv2.destroyAllWindows()