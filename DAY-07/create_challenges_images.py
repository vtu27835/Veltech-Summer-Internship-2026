import numpy as np
import cv2

# Create a main test image with shapes
main_img = np.zeros((400, 400, 3), dtype="uint8")

# Draw colorful shapes
cv2.rectangle(main_img, (50, 50), (150, 150), (255, 0, 0), -1)      # Blue square
cv2.circle(main_img, (300, 120), 60, (0, 255, 0), -1)                # Green circle
cv2.rectangle(main_img, (200, 250), (350, 350), (0, 0, 255), -1)     # Red square

cv2.line(main_img, (0, 0), (400, 400), (0, 255, 255), 5)             # Yellow diagonal
cv2.circle(main_img, (200, 200), 100, (255, 255, 255), 3)            # White circle outline

cv2.putText(
    main_img,
    "CHALLENGE",
    (100, 380),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.imwrite("challenge_main.png", main_img)

# Create a secondary image for blending
blend_img = np.zeros((400, 400, 3), dtype="uint8")

for y in range(400):
    color = int(y * 255 / 400)
    cv2.line(
        blend_img,
        (0, y),
        (400, y),
        (0, color, 255 - color),
        1
    )

cv2.imwrite("challenge_blend.png", blend_img)

print("Created challenge_main.png and challenge_blend.png")