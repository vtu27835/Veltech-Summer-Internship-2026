# exercise_2_color_picker.py

import cv2
import numpy as np


def color_picker():
    """Interactive color picker showing BGR and HSV values"""

    # Load image
    image = cv2.imread("test_image.png")

    if image is None:
        print("Create test_image.png first!")
        return

    # Create a blank canvas
    canvas = np.ones((400, 600, 3), dtype="uint8") * 255

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:

            # Get color at cursor
            if x < image.shape[1] and y < image.shape[0]:
                color = image[y, x]
            else:
                color = (0, 0, 0)

            b, g, r = color

            # Convert to HSV
            color_reshape = np.uint8([[color]])
            hsv_color = cv2.cvtColor(
                color_reshape,
                cv2.COLOR_BGR2HSV
            )[0][0]

            h, s, v = hsv_color

            # Create display copy
            display = canvas.copy()

            # Color swatch
            cv2.rectangle(
                display,
                (20, 20),
                (120, 120),
                (int(b), int(g), int(r)),
                -1
            )

            # Show values
            cv2.putText(
                display,
                f"BGR: ({b}, {g}, {r})",
                (140, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            cv2.putText(
                display,
                f"HSV: ({h}, {s}, {v})",
                (140, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            # Show image preview
            img_small = cv2.resize(image, (300, 200))
            display[200:400, 150:450] = img_small

            # Draw crosshair
            if x < image.shape[1] and y < image.shape[0]:
                cv2.drawMarker(
                    display,
                    (x + 150, y + 200),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    10,
                    2
                )

            cv2.imshow(
                "Color Picker (move mouse over image)",
                display
            )

    cv2.namedWindow("Color Picker")
    cv2.setMouseCallback("Color Picker", mouse_callback)

    print("Move mouse over the image to see color values")
    print("Press ESC to exit")

    # Initial display
    cv2.imshow(
        "Color Picker (move mouse over image)",
        canvas
    )

    while True:
        if cv2.waitKey(10) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()


color_picker()