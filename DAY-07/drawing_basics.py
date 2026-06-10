# drawing_basics.py
# Learning to draw lines, rectangles, circles, and text
# drawing_basics.py
from __future__ import print_function
import numpy as np
import cv2
import math

# --------------------------------------------------
# COLOR CONSTANTS (BGR FORMAT)
# --------------------------------------------------
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

print("=" * 60)
print("DAY 4: DRAWING ON IMAGES")
print("=" * 60)

# --------------------------------------------------
# SECTION 1: CREATING A CANVAS
# --------------------------------------------------
print("\n[Section 1] Creating a blank canvas...")

canvas = np.zeros((400, 400, 3), dtype="uint8")

cv2.imshow("Blank Canvas (Black)", canvas)
cv2.waitKey(0)

print("Created blank 400x400 black canvas")

white_canvas = np.ones((400, 400, 3), dtype="uint8") * 255

cv2.imshow("White Canvas", white_canvas)
cv2.waitKey(0)

print("Created white canvas")
# -----------------------------------------------------------------
# SECTION 2: DRAWING LINES
# -----------------------------------------------------------------
print("\n[Section 2] Drawing lines...")
# Reset canvas to black
canvas = np.zeros((400, 400, 3), dtype="uint8")
# Draw a green line from top-left to bottom-right
# Parameters: image, start_point, end_point, color, thickness
cv2.line(canvas, (0, 0), (400, 400), GREEN, 3)
cv2.imshow("Diagonal Green Line", canvas)
cv2.waitKey(0)
# Draw a red line from top-right to bottom-left
cv2.line(canvas, (400, 0), (0, 400), RED, 5)
cv2.imshow("Both Diagonals", canvas)
cv2.waitKey(0)
# Draw a horizontal blue line across the center
center_y = 200
cv2.line(canvas, (0, center_y), (400, center_y), BLUE, 2)
cv2.imshow("Added Horizontal Blue Line", canvas)
cv2.waitKey(0)
# Draw a vertical yellow line down the center
center_x = 200
cv2.line(canvas, (center_x, 0), (center_x, 400), YELLOW, 2)
cv2.imshow("Added Vertical Yellow Line", canvas)
cv2.waitKey(0)
# Create a fresh canvas for more line examples
canvas = np.zeros((400, 400, 3), dtype="uint8")
# Draw multiple lines with different thicknesses
for i in range(10):
 # Start from left edge, end at right edge
 y_pos = i * 40
 thickness = i + 1
 # Color changes with position
 color = (0, i * 25, 255 - i * 25) # B, G, R
 cv2.line(canvas, (0, y_pos), (400, y_pos), color, thickness)
cv2.imshow("Lines with Varying Thicknesses", canvas)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 3: DRAWING RECTANGLES
# -----------------------------------------------------------------
print("\n[Section 3] Drawing rectangles...")
# Reset canvas
canvas = np.zeros((400, 400, 3), dtype="uint8")
# Draw an outlined rectangle (top-left to bottom-right)
# Parameters: image, top-left, bottom-right, color, thickness
cv2.rectangle(canvas, (50, 50), (150, 150), GREEN, 3)
cv2.imshow("Outlined Rectangle", canvas)
cv2.waitKey(0)
# Draw another rectangle with different position
cv2.rectangle(canvas, (200, 50), (350, 150), BLUE, 5)
cv2.imshow("Two Outlined Rectangles", canvas)
cv2.waitKey(0)
# Draw a filled rectangle (thickness = -1 or cv2.FILLED)
cv2.rectangle(canvas, (50, 200), (150, 300), RED, -1)
cv2.imshow("Added Filled Red Rectangle", canvas)
cv2.waitKey(0)
# Draw a yellow filled rectangle
cv2.rectangle(canvas, (200, 200), (350, 300), YELLOW, cv2.FILLED)
cv2.imshow("All Rectangles", canvas)
cv2.waitKey(0)
# Create a pattern of rectangles
canvas = np.zeros((400, 400, 3), dtype="uint8")
for i in range(8):
 size = 50
 x = i * 50
 y = 0
 color = (i * 32, 255 - i * 32, 128)
 cv2.rectangle(canvas, (x, y), (x + size, 400), color, -1)
cv2.imshow("Striped Pattern", canvas)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 4: DRAWING CIRCLES
# -----------------------------------------------------------------
print("\n[Section 4] Drawing circles...")
# Reset canvas
canvas = np.zeros((400, 400, 3), dtype="uint8")
# Get center of canvas
center_x = canvas.shape[1] // 2
center_y = canvas.shape[0] // 2
# Draw a single circle at center
# Parameters: image, center, radius, color, thickness
cv2.circle(canvas, (center_x, center_y), 50, WHITE, 3)
cv2.imshow("Single Circle", canvas)
cv2.waitKey(0)
# Draw concentric circles (like a bullseye)
canvas = np.zeros((400, 400, 3), dtype="uint8")
for radius in range(20, 200, 20):
    if radius % 40 == 20:
        color = RED
    else:
        color = WHITE

    cv2.circle(canvas, (center_x, center_y), radius, color, 3)
cv2.imshow("Bullseye (Concentric Circles)", canvas)
cv2.waitKey(0)
# Draw a solid circle (filled)
canvas = np.zeros((400, 400, 3), dtype="uint8")
cv2.circle(canvas, (center_x, center_y), 100, BLUE, -1)
cv2.imshow("Filled Circle", canvas)
cv2.waitKey(0)
# Draw multiple random circles (preview of what's coming)
print("Drawing 25 random circles...")
canvas = np.zeros((400, 400, 3), dtype="uint8")
for i in range(25):
 # Random radius between 10 and 80
 radius = np.random.randint(10, 80)
 
 # Random color (B, G, R each 0-255)
 color = np.random.randint(0, 256, size=(3,)).tolist()
 
 # Random position
 x = np.random.randint(radius, 400 - radius)
 y = np.random.randint(radius, 400 - radius)
 
 # Random thickness (-1 = filled, or 1-10 for outline)
 thickness = np.random.choice([-1, 1, 2, 3])
 
 cv2.circle(canvas, (x, y), radius, color, thickness)
cv2.imshow("Random Circles", canvas)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 5: ADDING TEXT
# -----------------------------------------------------------------
print("\n[Section 5] Adding text to images...")
# Reset canvas
canvas = np.zeros((400, 400, 3), dtype="uint8")
# Add text
# Parameters: image, text, position, font, scale, color, thickness
cv2.putText(canvas, "Hello OpenCV!", (50, 100),
 cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
cv2.imshow("Text on Image", canvas)
cv2.waitKey(0)
# Add multiple texts with different styles
canvas = np.zeros((500, 600, 3), dtype="uint8")
# Different fonts
fonts = [
 (cv2.FONT_HERSHEY_SIMPLEX, "SIMPLEX"),
 (cv2.FONT_HERSHEY_PLAIN, "PLAIN"),
 (cv2.FONT_HERSHEY_DUPLEX, "DUPLEX"),
 (cv2.FONT_HERSHEY_COMPLEX, "COMPLEX"),
 (cv2.FONT_HERSHEY_TRIPLEX, "TRIPLEX"),
 (cv2.FONT_HERSHEY_COMPLEX_SMALL, "COMPLEX_SMALL"),
 (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, "SCRIPT_SIMPLEX"),
 (cv2.FONT_HERSHEY_SCRIPT_COMPLEX, "SCRIPT_COMPLEX")
]
for i, (font, name) in enumerate(fonts):
 y_pos = 40 + i * 50
 cv2.putText(canvas, name, (20, y_pos), font, 0.8, WHITE, 2)
cv2.imshow("Different Fonts", canvas)
cv2.waitKey(0)
# Text with different sizes and colors
canvas = np.zeros((400, 600, 3), dtype="uint8")
texts = [
 ("Small text", 0.5, (50, 60), GREEN),
 ("Normal text", 1.0, (50, 120), BLUE),
 ("Large text", 1.5, (50, 190), RED),
 ("Extra Large", 2.0, (50, 280), YELLOW)
]
for text, scale, position, color in texts:
 cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
 scale, color, 2)
cv2.imshow("Text Sizes", canvas)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 6: COMBINING EVERYTHING - MASTERPIECE
# -----------------------------------------------------------------
print("\n[Section 6] Creating a masterpiece combining all skills...")
# Create a canvas for our masterpiece
masterpiece = np.zeros((500, 500, 3), dtype="uint8")
center_x, center_y = 250, 250
# 1. Draw a colorful background (gradient effect)
for y in range(500):
    color_value = int(y * 255 / 500)

    cv2.line(
        masterpiece,
        (0, y),
        (500, y),
        (0, color_value, 255 - color_value),
        1
    )
# 2. Draw a large rectangle border
cv2.rectangle(masterpiece, (50, 50), (450, 450), WHITE, 3)
# 3. Draw concentric circles in the center
for r in range(30, 200, 30):
 color = (r * 5 % 256, (r * 10) % 256, (r * 15) % 256)
 cv2.circle(masterpiece, (center_x, center_y), r, color, 2)
# 4. Draw a star pattern (diagonal lines)
for angle in range(0, 360, 30):
 import math
 rad = math.radians(angle)
 x2 = int(center_x + 200 * math.cos(rad))
 y2 = int(center_y + 200 * math.sin(rad))
 cv2.line(masterpiece, (center_x, center_y), (x2, y2), CYAN, 2)
# 5. Add text
cv2.putText(masterpiece, "Computer Vision", (120, 470),
 cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)
cv2.putText(masterpiece, "Day 4: Drawing", (160, 50),
 cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
cv2.imshow("MASTERPIECE", masterpiece)
cv2.waitKey(0)
# Save your masterpiece
cv2.imwrite("my_masterpiece.png", masterpiece)
print("Masterpiece saved as 'my_masterpiece.png'")
print("\n" + "=" * 60)
print("DAY 4 COMPLETE!")
print("=" * 60)
cv2.destroyAllWindows()