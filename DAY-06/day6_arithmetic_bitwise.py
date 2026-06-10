# day6_arithmetic_bitwise.py
# Complete guide to image arithmetic and bitwise operations
from __future__ import print_function
import numpy as np
import argparse
import cv2
print("=" * 70)
print("DAY 6: IMAGE ARITHMETIC & BITWISE OPERATIONS")
print("=" * 70)
# -----------------------------------------------------------------
# SECTION 1: UNDERSTANDING OPENCV VS NUMPY ARITHMETIC
# -----------------------------------------------------------------
print("\n[Section 1] OpenCV vs NumPy Arithmetic")
print("-" * 50)
# Create simple test values
value1 = np.uint8([200])
value2 = np.uint8([100])
print(f"Test values: {value1[0]} + {value2[0]} = ?")
print(f"Test values: {value1[0]} - {value2[0]} = ?")
print("\n--- OpenCV (Clipping) ---")
cv2_add = cv2.add(value1, value2)
cv2_sub = cv2.subtract(value1, value2)
print(f"cv2.add(200, 100) = {cv2_add[0]} (clipped to 255)")
print(f"cv2.subtract(200, 100) = {cv2_sub[0]}")
print("\n--- NumPy (Modulo/Wrap Around) ---")
np_add = value1 + value2
np_sub = value1 - value2
print(f"np.uint8(200) + np.uint8(100) = {np_add[0]} (wrapped around!)")
print(f"np.uint8(200) - np.uint8(100) = {np_sub[0]} (wrapped around!)")
print("\n--- More Examples ---")
test_values = [(250, 20), (10, 20), (255, 1), (0, 1)]
for v1, v2 in test_values:
    a = np.uint8([v1])
    b = np.uint8([v2])
    print(f"OpenCV: {v1} + {v2} = {cv2.add(a,b)[0]}")
    print(f"NumPy: {v1} + {v2} = {(a+b)[0]}")
    print(f"OpenCV: {v1} - {v2} = {cv2.subtract(a,b)[0]}")
    print(f"NumPy: {v1} - {v2} = {(a-b)[0]}")
    print("-" * 30)
# -----------------------------------------------------------------
# SECTION 2: LOAD IMAGES FOR ARITHMETIC
# -----------------------------------------------------------------
print("\n[Section 2] Loading images for arithmetic operations...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="Path to the first image")
ap.add_argument("-j", "--image2", required=True,help="Path to the second image (for blending)")
args = vars(ap.parse_args())
image1 = cv2.imread(args["image"])
image2 = cv2.imread(args["image2"])
if image1 is None or image2 is None:
    print("ERROR: Could not load one or both images!")
    exit()
# Resize images to same dimensions if needed
if image1.shape != image2.shape:
    print(f"Images have different sizes: {image1.shape} vs {image2.shape}")
    print("Resizing image2 to match image1...")
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    height, width = image1.shape[:2]
print(f"Both images now: {width} x {height} pixels")
cv2.imshow("Image 1", image1)
cv2.waitKey(0)
cv2.imshow("Image 2", image2)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 3: ADDITION OPERATIONS
# -----------------------------------------------------------------
print("\n[Section 3] Image Addition")
print("-" * 50)
# Add a constant to every pixel (makes image brighter)
print("Adding 100 to every pixel...")
M = np.ones(image1.shape, dtype="uint8") * 100
brighter = cv2.add(image1, M)
cv2.imshow("Original", image1)
cv2.imshow("Brighter (+100 to all pixels)", brighter)
cv2.waitKey(0)
# Add two images together
print("Adding two images together...")
added = cv2.add(image1, image2)
cv2.imshow("Image 1 + Image 2 (OpenCV clipping)", added)
cv2.waitKey(0)
# NumPy addition (wrap around - creates interesting effects)
np_added = image1 + image2
cv2.imshow("Image 1 + Image 2 (NumPy wrap around)", np_added)
cv2.waitKey(0)
# Create comparison
comparison = np.hstack([image1, image2, added, np_added])
cv2.imshow("Comparison: Img1 | Img2 | OpenCV Add | NumPy Add",
 cv2.resize(comparison, (width*2, height//2)))
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 4: SUBTRACTION OPERATIONS
# -----------------------------------------------------------------
print("\n[Section 4] Image Subtraction")
print("-" * 50)
# Subtract constant (makes image darker)
print("Subtracting 80 from every pixel...")
M = np.ones(image1.shape, dtype="uint8") * 80
darker = cv2.subtract(image1, M)
cv2.imshow("Original", image1)
cv2.imshow("Darker (-80 from all pixels)", darker)
cv2.waitKey(0)
# Subtract image2 from image1
subtracted = cv2.subtract(image1, image2)
cv2.imshow("Image 1 - Image 2 (differences highlighted)", subtracted)
cv2.waitKey(0)
# Absolute difference (shows magnitude of difference regardless of direction)
abs_diff = cv2.absdiff(image1, image2)
cv2.imshow("Absolute Difference (highlights where images differ)",abs_diff)
cv2.waitKey(0)
# Create grayscale version of difference for better visualization
gray_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
cv2.imshow("Difference Intensity (white = different)", gray_diff)
cv2.waitKey(0)
print("Observation: White areas in difference images show where images differ most!")
# -----------------------------------------------------------------
# SECTION 5: IMAGE BLENDING
# -----------------------------------------------------------------
print("\n[Section 5] Image Blending")
print("-" * 50)
# Method 1: Manual weighted addition
alpha = 0.5 # Weight for first image
beta = 0.5 # Weight for second image
blended_manual = cv2.addWeighted(image1, alpha, image2, beta, 0)
cv2.imshow(f"Blended 50%-50% (Manual)", blended_manual)
cv2.waitKey(0)
# Method 2: Using cv2.addWeighted (built-in)
print("Using cv2.addWeighted() for different blend ratios...")
# Try different blend ratios
blends = [
 (0.3, 0.7, "30% Image1, 70% Image2"),
 (0.5, 0.5, "50% Image1, 50% Image2"),
 (0.7, 0.3, "70% Image1, 30% Image2"),
 (0.9, 0.1, "90% Image1, 10% Image2")
]
for alpha, beta, label in blends:
    blended = cv2.addWeighted(image1, alpha, image2, beta, 0)
    cv2.putText(blended, label, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(label, blended)
    cv2.waitKey(0)
# Create a smooth transition (gradient blend)
print("\n--- Gradient Blend (Fade Effect) ---")
gradient_blend = image1.copy()
for x in range(width):
 # Alpha increases from left (0) to right (1)
    alpha = x / width
    beta = 1 - alpha
 
 # Blend each column independently
    gradient_blend[:, x:x+1] = cv2.addWeighted(
    image1[:, x:x+1], alpha,
    image2[:, x:x+1], beta, 0
    )
cv2.imshow("Gradient Blend (smooth transition left to right)",gradient_blend)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 6: BITWISE OPERATIONS (WITH SHAPES FOR CLARITY)
# -----------------------------------------------------------------
print("\n[Section 6] Bitwise Operations")
print("-" * 50)
# Create simple shapes to demonstrate bitwise operations
print("Creating shapes for demonstration...")
rectangle = np.zeros((300, 300), dtype="uint8")
circle = np.zeros((300, 300), dtype="uint8")
# Draw a white rectangle (center area)
cv2.rectangle(rectangle, (50, 50), (250, 250), 255, -1)
# Draw a white circle (center area)
cv2.circle(circle, (150, 150), 120, 255, -1)
cv2.imshow("Rectangle Shape", rectangle)
cv2.imshow("Circle Shape", circle)
cv2.waitKey(0)
print("\n--- AND Operation ---")
# AND: True only if BOTH pixels are white
and_result = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND: Rectangle AND Circle (overlap only)", and_result)
cv2.waitKey(0)
print("\n--- OR Operation ---")
# OR: True if EITHER pixel is white
or_result = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR: Rectangle OR Circle (union)", or_result)
cv2.waitKey(0)
print("\n--- XOR Operation ---")
# XOR: True if ONE (but not both) is white
xor_result = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR: Rectangle XOR Circle (exclusive or)", xor_result)
cv2.waitKey(0)
print("\n--- NOT Operation ---")
not_rectangle = cv2.bitwise_not(rectangle)
not_circle = cv2.bitwise_not(circle)
cv2.imshow("NOT Rectangle (inverted)", not_rectangle)
cv2.imshow("NOT Circle (inverted)", not_circle)
cv2.waitKey(0)
# Side-by-side comparison
top_row = np.hstack([rectangle, circle, and_result])
bottom_row = np.hstack([or_result, xor_result, not_rectangle])
comparison = np.vstack([top_row, bottom_row])
cv2.imshow("Bitwise Operations Summary", comparison)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 7: APPLYING BITWISE TO COLOR IMAGES
# -----------------------------------------------------------------
print("\n[Section 7] Bitwise Operations on Color Images")
print("-" * 50)
# Create a mask (white rectangle)
mask = np.zeros(image1.shape[:2], dtype="uint8")
cv2.rectangle(mask, (width//4, height//4), (3*width//4, 3*height//4),255, -1)
cv2.imshow("Mask (white rectangle)", mask)
cv2.waitKey(0)
# Apply mask using bitwise AND
masked = cv2.bitwise_and(image1, image1, mask=mask)
cv2.imshow("Image with Mask Applied (only rectangle visible)", masked)
cv2.waitKey(0)
# Create a circular mask
circle_mask = np.zeros(image1.shape[:2], dtype="uint8")
cv2.circle(circle_mask, (width//2, height//2), min(width, height)//3,255, -1)
cv2.imshow("Circular Mask", circle_mask)
cv2.waitKey(0)
# Apply circular mask
circle_masked = cv2.bitwise_and(image1, image1, mask=circle_mask)
cv2.imshow("Image with Circular Mask", circle_masked)
cv2.waitKey(0)
# Create an inverted mask (everything EXCEPT the rectangle)
inverted_mask = cv2.bitwise_not(mask)
inverted_masked = cv2.bitwise_and(image1, image1, mask=inverted_mask)
cv2.imshow("Inverted Mask (everything EXCEPT rectangle)",inverted_masked)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 8: PRACTICAL APPLICATIONS
# -----------------------------------------------------------------
print("\n[Section 8] Practical Applications")
print("-" * 50)
# Application 1: Extract specific color region
print("Application 1: Extracting a specific color region")
# Create a color range mask (for demonstration - we'll learn more later)
hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
# For now, just use a rectangle mask
region_mask = np.zeros(image1.shape[:2], dtype="uint8")
cv2.rectangle(region_mask, (width//3, height//3), (2*width//3,2*height//3), 255, -1)
region_extracted = cv2.bitwise_and(image1, image1, mask=region_mask)
cv2.imshow("Extracted Center Region", region_extracted)
cv2.waitKey(0)
# Application 2: Create a vignette effect
print("Application 2: Creating a vignette effect...")
vignette_mask = np.zeros(image1.shape[:2], dtype="uint8")
center_x, center_y = width//2, height//2
for y in range(height):
    for x in range(width):
 # Calculate distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
 # Pixel value decreases with distance from center
        value = int(255 * (1 - dist / max_dist))
        vignette_mask[y, x] = value
vignette = cv2.bitwise_and(image1, image1, mask=vignette_mask)
cv2.imshow("Vignette Effect (dark edges)", vignette)
cv2.waitKey(0)
# Application 3: Combine two images using a mask
print("Application 3: Combining two images with a mask...")
# Create a gradient mask
gradient_mask = np.zeros(image1.shape[:2], dtype="uint8")
for x in range(width):
    value = int(255 * x / width)
    gradient_mask[:, x] = value
# Where mask is white, show image1; where black, show image2
combined = np.where(gradient_mask[:, :, np.newaxis] > 128, image1,image2)
cv2.imshow("Gradient Mask", gradient_mask)
cv2.imshow("Combined: Left=Image2, Right=Image1", combined)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 9: CREATE A COMPOSITE MASTERPIECE
# -----------------------------------------------------------------
print("\n[Section 9] Challenge: Create a Composite Image")
print("-" * 50)
# Create a blank canvas
composite = np.zeros((height, width, 3), dtype="uint8")
# Place image1 in the center region using mask
center_mask = np.zeros((height, width), dtype="uint8")
cv2.rectangle(center_mask, (width//4, height//4), (3*width//4,3*height//4), 255, -1)
composite = cv2.bitwise_and(composite, composite) # Start black
temp = cv2.bitwise_and(image1, image1, mask=center_mask)
composite = cv2.add(composite, temp)
# Add image2 around the edges using inverted mask
edge_mask = cv2.bitwise_not(center_mask)
temp2 = cv2.bitwise_and(image2, image2, mask=edge_mask)
composite = cv2.add(composite, temp2)
# Add a border
cv2.rectangle(composite, (0, 0), (width-1, height-1), (0, 255, 0), 10)
cv2.imshow("COMPOSITE: Image1 in Center, Image2 on Edges", composite)
cv2.waitKey(0)
cv2.imwrite("composite_masterpiece.png", composite)
# -----------------------------------------------------------------
# SECTION 10: COMPARISON SUMMARY
# -----------------------------------------------------------------
print("\n[Section 10] Summary Comparison")
# Create a comparison grid of all operations
operations = {
 "Original 1": image1,
 "Original 2": image2,
 "Addition": brighter,
 "Subtraction": darker,
 "Blended 50-50": blended_manual,
 "AND Masked": masked,
 "XOR Shapes": cv2.cvtColor(xor_result, cv2.COLOR_GRAY2BGR) if
len(xor_result.shape) == 2 else xor_result
}
# Create a grid (2x4)
grid_rows = []
current_row = []
for i, (name, img) in enumerate(operations.items()):
 # Ensure image is color for display
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
 
 # Resize to consistent height
        img = cv2.resize(img, (200, 150))
 
 # Add label
        cv2.putText(img, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255, 255), 1)
        current_row.append(img)
 
    if len(current_row) == 4:
        grid_rows.append(np.hstack(current_row))
        current_row = []
if current_row:
    while len(current_row) < 4:
        blank = np.zeros((150, 200, 3), dtype="uint8")
        current_row.append(blank)
    grid_rows.append(np.hstack(current_row)) 
summary_grid = np.vstack(grid_rows)
cv2.imshow("SUMMARY: All Operations Compared", summary_grid)
cv2.waitKey(0)
cv2.imwrite("day6_summary.png", summary_grid)
print("\n" + "=" * 70)
print("DAY 6 COMPLETE!")
print("=" * 70)
cv2.destroyAllWindows()