# day13_contours.py 
# Complete guide to contours in OpenCV
from __future__ import print_function 
import numpy as np 
import argparse 
import cv2

print("=" * 70) 
print("DAY 13: CONTOURS") 
print("=" * 70)

# ----------------------------------------------------------------- 
# SECTION 1: LOAD IMAGE AND CREATE BINARY IMAGE 
# ----------------------------------------------------------------- 
print("\n[Section 1] Loading image and creating binary image...")

ap = argparse.ArgumentParser() 
ap.add_argument("-i", "--image", required=True, help="Path to the image") 
args = vars(ap.parse_args())

image = cv2.imread(args["image"]) 
if image is None: 
    print("ERROR: Could not load image!") 
    exit()

height, width = image.shape[:2] 
print(f"Loaded: {width} x {height} pixels")

# Convert to grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create binary image using thresholding (Contours work on binary images) 
blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Original Image", image) 
cv2.imshow("Binary Image (for contours)", binary) 
cv2.waitKey(0)

# Create a test image with known shapes for demonstration 
print("\nCreating test image with known shapes...") 
test_shapes = np.zeros((400, 600, 3), dtype="uint8")

# Draw different shapes 
cv2.rectangle(test_shapes, (20, 20), (120, 120), (200, 200, 200), -1) 
cv2.circle(test_shapes, (250, 70), 50, (200, 200, 200), -1) 
cv2.ellipse(test_shapes, (400, 70), (60, 40), 0, 0, 360, (200, 200, 200), -1) 
cv2.rectangle(test_shapes, (20, 200), (120, 350), (200, 200, 200), -1) 
cv2.circle(test_shapes, (250, 275), 70, (200, 200, 200), -1) 
cv2.ellipse(test_shapes, (450, 275), (80, 50), 30, 0, 360, (200, 200, 200), -1)

# Add some text 
cv2.putText(test_shapes, "TEST SHAPES", (200, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imshow("Test Shapes", test_shapes) 
cv2.waitKey(0)

# Convert test shapes to binary 
gray_shapes = cv2.cvtColor(test_shapes, cv2.COLOR_BGR2GRAY) 
_, binary_shapes = cv2.threshold(gray_shapes, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary Test Shapes", binary_shapes) 
cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 2: FINDING CONTOURS 
# ----------------------------------------------------------------- 
print("\n[Section 2] Finding Contours") 
print("-" * 50) 
print("Contours find connected boundaries in binary images.") 
print("OpenCV 3.0+ returns: (image, contours, hierarchy)")

# Find contours in binary image 
# Using RETR_EXTERNAL to get only outermost contours 
print("\n--- Finding contours on test shapes ---")

contours, hierarchy = cv2.findContours(binary_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:] # Ignore the first return value for compatibility

print(f"Found {len(contours)} contours")

# Draw all contours 
contour_vis = test_shapes.copy() 
cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 3)

cv2.imshow("Test Shapes with Contours (Green)", contour_vis) 
cv2.waitKey(0)

print("\n--- Different Retrieval Modes ---")

# RETR_EXTERNAL: Only outer contours 
contours_external, _ = cv2.findContours(binary_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:]

# RETR_LIST: All contours 
contours_list, _ = cv2.findContours(binary_shapes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1:]

# RETR_TREE: Full hierarchy 
contours_tree, hierarchy_tree = cv2.findContours(binary_shapes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]

print(f"RETR_EXTERNAL: {len(contours_external)} contours") 
print(f"RETR_LIST: {len(contours_list)} contours") 
print(f"RETR_TREE: {len(contours_tree)} contours")

# Visualize different retrieval modes 
vis_external = test_shapes.copy() 
vis_list = test_shapes.copy() 
vis_tree = test_shapes.copy()

cv2.drawContours(vis_external, contours_external, -1, (0, 255, 0), 3) 
cv2.drawContours(vis_list, contours_list, -1, (0, 255, 0), 3) 
cv2.drawContours(vis_tree, contours_tree, -1, (0, 255, 0), 3)

cv2.putText(vis_external, f"RETR_EXTERNAL: {len(contours_external)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
cv2.putText(vis_list, f"RETR_LIST: {len(contours_list)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
cv2.putText(vis_tree, f"RETR_TREE: {len(contours_tree)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

comparison = np.hstack([vis_external, vis_list, vis_tree]) 
cv2.imshow("Contour Retrieval Modes: EXTERNAL | LIST | TREE", comparison) 
cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 3: CONTOUR PROPERTIES 
# ----------------------------------------------------------------- 
print("\n[Section 3] Analyzing Contour Properties") 
print("-" * 50)

# Find contours on real image 
contours_real, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:]

print(f"Found {len(contours_real)} contours in your image")

# Analyze each contour 
print("\nContour Properties:") 
print("-" * 40)

for i, contour in enumerate(contours_real): 
    # Area 
    area = cv2.contourArea(contour)
    
    # Perimeter 
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box 
    x, y, w, h = cv2.boundingRect(contour)
    
    # Center (using moments) 
    M = cv2.moments(contour) 
    if M["m00"] != 0: 
        cx = int(M["m10"] / M["m00"]) 
        cy = int(M["m01"] / M["m00"]) 
    else: 
        cx, cy = x + w//2, y + h//2
        
    # Print properties 
    print(f"Contour {i+1}:") 
    print(f" Area: {area:.0f} pixels") 
    print(f" Perimeter: {perimeter:.1f} pixels") 
    print(f" Bounding Box: ({x}, {y}) - {w}×{h}") 
    print(f" Center: ({cx}, {cy})") 
    print("-" * 40)

# Visualize contour properties 
prop_vis = image.copy()

for i, contour in enumerate(contours_real): 
    # Draw contour 
    cv2.drawContours(prop_vis, [contour], -1, (0, 255, 0), 2)
    
    # Draw bounding box 
    x, y, w, h = cv2.boundingRect(contour) 
    cv2.rectangle(prop_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Draw center 
    M = cv2.moments(contour) 
    if M["m00"] != 0: 
        cx = int(M["m10"] / M["m00"]) 
        cy = int(M["m01"] / M["m00"]) 
        cv2.circle(prop_vis, (cx, cy), 5, (0, 0, 255), -1)
        
    # Label contour number 
    cv2.putText(prop_vis, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

cv2.imshow("Contour Properties: Green=Contour, Blue=Box, Red=Center", prop_vis)
cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 4: DRAWING INDIVIDUAL CONTOURS 
# ----------------------------------------------------------------- 
print("\n[Section 4] Drawing Individual Contours") 
print("-" * 50)

# Draw each contour one at a time 
print("Drawing each contour individually:")

for i, contour in enumerate(contours_real): 
    vis = image.copy() 
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3) 
    cv2.putText(vis, f"Contour {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
    cv2.imshow(f"Contour {i+1}", vis) 
    cv2.waitKey(0)

# Draw contours with different colors 
color_vis = image.copy() 
for i, contour in enumerate(contours_real): 
    # Generate random color 
    color = np.random.randint(0, 256, 3).tolist() 
    cv2.drawContours(color_vis, [contour], -1, color, 3)

cv2.imshow("All Contours with Different Colors", color_vis) 
cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 5: FILTERING CONTOURS BY AREA 
# ----------------------------------------------------------------- 
print("\n[Section 5] Filtering Contours by Area") 
print("-" * 50)

print("Filtering out small contours (noise)...")

min_area = 100 # Minimum area to keep 
filtered_contours = [c for c in contours_real if cv2.contourArea(c) > min_area]

print(f"Original: {len(contours_real)} contours") 
print(f"Filtered (area > {min_area}): {len(filtered_contours)} contours")

# Visualize filtering 
filtered_vis = image.copy()

for contour in filtered_contours: 
    cv2.drawContours(filtered_vis, [contour], -1, (0, 255, 0), 3)

cv2.imshow(f"Filtered Contours (area > {min_area})", filtered_vis) 
cv2.waitKey(0)

# Try different area thresholds 
print("\n--- Different Area Thresholds ---") 
for threshold in [50, 200, 500, 1000]: 
    filtered = [c for c in contours_real if cv2.contourArea(c) > threshold] 
    print(f"Area > {threshold}: {len(filtered)} contours")

# -----------------------------------------------------------------
# SECTION 6: CONTOUR APPROXIMATION 
# ----------------------------------------------------------------- 
print("\n[Section 6] Contour Approximation") 
print("-" * 50) 
print("Approximation simplifies contours by reducing points.") 
print("Useful for shape detection (triangles, rectangles, circles)")

# Get a contour from test shapes (the large rectangle) 
test_contours, _ = cv2.findContours(binary_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:]

# Take the first rectangle contour (should be the rectangle) 
rect_contour = test_contours[0] 
print(f"Original contour points: {len(rect_contour)}")

# Approximate with different epsilon values 
epsilons = [0.01, 0.02, 0.05, 0.1]

for eps in epsilons: 
    perimeter = cv2.arcLength(rect_contour, True) 
    approx = cv2.approxPolyDP(rect_contour, eps * perimeter, True)
    print(f"Epsilon={eps:.2f}: {len(approx)} points")
    
    # Visualize 
    vis = test_shapes.copy() 
    cv2.drawContours(vis, [approx], -1, (0, 255, 0), 3) 
    cv2.putText(vis, f"Approx: {len(approx)} points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
    cv2.imshow(f"Approximation - eps={eps}", vis) 
    cv2.waitKey(0)

print("\nObservation: Higher epsilon = more simplification") 
print(" - Too high = loses shape details") 
print(" - Too low = too many points")

# ----------------------------------------------------------------- 
# SECTION 7: SHAPE DETECTION USING CONTOURS 
# ----------------------------------------------------------------- 
print("\n[Section 7] Shape Detection") 
print("-" * 50)

def detect_shape(contour): 
    """Detect the shape type from a contour""" 
    # Approximate contour 
    perimeter = cv2.arcLength(contour, True) 
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    # Number of vertices determines shape 
    vertices = len(approx)
    
    if vertices == 3: 
        return "Triangle" 
    elif vertices == 4: 
        # Check if it's a square or rectangle 
        x, y, w, h = cv2.boundingRect(approx) 
        aspect_ratio = w / h 
        if 0.95 <= aspect_ratio <= 1.05: 
            return "Square" 
        else:
            return "Rectangle" 
    elif vertices == 5: 
        return "Pentagon" 
    elif vertices == 6: 
        return "Hexagon" 
    elif vertices > 6: 
        return "Circle" 
    else: 
        return "Unknown"

# Detect shapes in test image 
shape_vis = test_shapes.copy() 
test_contours, _ = cv2.findContours(binary_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:]

for contour in test_contours: 
    shape_name = detect_shape(contour)
    
    # Draw contour 
    cv2.drawContours(shape_vis, [contour], -1, (0, 255, 0), 2)
    
    # Label shape 
    M = cv2.moments(contour) 
    if M["m00"] != 0: 
        cx = int(M["m10"] / M["m00"]) 
        cy = int(M["m01"] / M["m00"]) 
        cv2.putText(shape_vis, shape_name, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Shape Detection Results", shape_vis) 
cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 8: EXTRACTING OBJECTS USING CONTOURS 
# ----------------------------------------------------------------- 
print("\n[Section 8] Extracting Objects Using Contours") 
print("-" * 50)

# Extract each object from the image 
print("Extracting objects using bounding boxes...")

for i, contour in enumerate(contours_real[:5]): # Limit to first 5 
    # Get bounding box 
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract the object 
    obj = image[y:y+h, x:x+w]
    
    # Draw bounding box on original 
    extracted_vis = image.copy() 
    cv2.rectangle(extracted_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    cv2.imshow(f"Extracted Object {i+1}", obj) 
    cv2.imshow(f"Object {i+1} Location", extracted_vis) 
    cv2.waitKey(0)

print("\n--- Using Circles (minEnclosingCircle) ---")

for i, contour in enumerate(contours_real[:5]): 
    # Fit circle to contour
    if len(contour) > 4: # Need at least 5 points for circle 
        (cx, cy), radius = cv2.minEnclosingCircle(contour) 
        center = (int(cx), int(cy)) 
        radius = int(radius)
        
        # Draw circle 
        vis = image.copy() 
        cv2.circle(vis, center, radius, (0, 255, 0), 3) 
        cv2.putText(vis, f"Circle {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        cv2.imshow(f"Enclosing Circle {i+1}", vis) 
        cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 9: CONTOUR HIERARCHY (NESTED OBJECTS) 
# ----------------------------------------------------------------- 
print("\n[Section 9] Contour Hierarchy - Nested Objects") 
print("-" * 50)

# Create an image with nested shapes 
nested = np.zeros((400, 400, 3), dtype="uint8")

# Outer shape (large square) 
cv2.rectangle(nested, (50, 50), (350, 350), (200, 200, 200), -1)

# Inner shapes 
cv2.circle(nested, (200, 200), 60, (255, 255, 255), -1) 
cv2.rectangle(nested, (150, 150), (250, 250), (128, 128, 128), -1) 
cv2.circle(nested, (200, 200), 30, (200, 200, 200), -1)

cv2.imshow("Nested Shapes", nested) 
cv2.waitKey(0)

# Convert to binary 
gray_nested = cv2.cvtColor(nested, cv2.COLOR_BGR2GRAY) 
_, binary_nested = cv2.threshold(gray_nested, 127, 255, cv2.THRESH_BINARY)

# Find contours with hierarchy 
contours_nested, hierarchy_nested = cv2.findContours(binary_nested, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:]

print(f"Found {len(contours_nested)} contours") 
print(f"Hierarchy shape: {hierarchy_nested.shape}") 
print("\nHierarchy information (for each contour):") 
print("[Next, Previous, First_Child, Parent]") 
for i, h in enumerate(hierarchy_nested[0]): 
    print(f"Contour {i}: {h}")

# Visualize hierarchy 
hier_vis = nested.copy()

# Draw contours with different colors based on level 
for i, contour in enumerate(contours_nested): 
    color = (0, 255, 0) if hierarchy_nested[0][i][3] == -1 else (0, 0, 255) 
    cv2.drawContours(hier_vis, [contour], -1, color, 3)
    
    # Label 
    M = cv2.moments(contour)
    if M["m00"] != 0: 
        cx = int(M["m10"] / M["m00"]) 
        cy = int(M["m01"] / M["m00"]) 
        parent = hierarchy_nested[0][i][3] 
        cv2.putText(hier_vis, str(i), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("Contour Hierarchy: Green=Outer, Red=Inner", hier_vis) 
cv2.waitKey(0)

# ----------------------------------------------------------------- 
# SECTION 10: PRACTICAL APPLICATION - COUNTING COINS 
# ----------------------------------------------------------------- 
print("\n[Section 10] Practical Application: Counting Coins") 
print("-" * 50)

# Create a simulated coin image 
coins = np.zeros((400, 500, 3), dtype="uint8")

# Draw coins (circles) with different sizes 
coin_positions = [ 
    (100, 100, 35), (200, 80, 30), (300, 120, 40), 
    (150, 200, 25), (250, 220, 35), (350, 200, 30), 
    (100, 300, 30), (200, 320, 25), (300, 310, 35), 
    (400, 300, 40) 
]

for x, y, r in coin_positions: 
    cv2.circle(coins, (x, y), r, (200, 200, 200), -1)

# Add some noise 
noise = np.random.randint(0, 30, coins.shape, dtype="uint8") 
coins = cv2.add(coins, noise)

cv2.imshow("Coins Image", coins) 
cv2.waitKey(0)

# Process coins 
gray_coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY) 
blurred_coins = cv2.GaussianBlur(gray_coins, (5, 5), 0) 
_, binary_coins = cv2.threshold(blurred_coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours 
coin_contours, _ = cv2.findContours(binary_coins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:]

print(f"Found {len(coin_contours)} coins!")

# Draw and label coins 
coin_result = coins.copy() 
for i, contour in enumerate(coin_contours): 
    # Draw contour 
    cv2.drawContours(coin_result, [contour], -1, (0, 255, 0), 2)
    
    # Calculate area (for coin size estimation) 
    area = cv2.contourArea(contour)
    
    # Label 
    M = cv2.moments(contour) 
    if M["m00"] != 0: 
        cx = int(M["m10"] / M["m00"]) 
        cy = int(M["m01"] / M["m00"]) 
        cv2.putText(coin_result, f"#{i+1}", (cx-15, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.putText(coin_result, f"Total Coins: {len(coin_contours)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
cv2.imshow("Coins Detected and Counted", coin_result) 
cv2.waitKey(0) 
cv2.imwrite("coins_detected.png", coin_result)

print("\nCoin Detection Pipeline:") 
print(" 1. Convert to grayscale ✓") 
print(" 2. Blur to reduce noise ✓") 
print(" 3. Otsu thresholding ✓") 
print(" 4. Find contours ✓") 
print(" 5. Count and label ✓")

# ----------------------------------------------------------------- 
# SECTION 11: CREATE CONTOUR REFERENCE GUIDE 
# ----------------------------------------------------------------- 
print("\n[Section 11] Creating Contour Reference Guide")

reference = np.zeros((650, 800, 3), dtype="uint8") 
ref_height, ref_width = reference.shape[:2]

# Title 
cv2.putText(reference, "CONTOUR REFERENCE GUIDE", (ref_width//2 - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Example contours visualization 
contour_example = np.zeros((150, 150), dtype="uint8") 
cv2.circle(contour_example, (75, 75), 60, 255, -1) 
contours_ex, _ = cv2.findContours(contour_example, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1:]

if contours_ex: 
    # Draw contour 
    contour_display = cv2.cvtColor(contour_example, cv2.COLOR_GRAY2BGR) 
    cv2.drawContours(contour_display, contours_ex, -1, (0, 255, 0), 3) 
    contour_display = cv2.resize(contour_display, (150, 150)) 
    reference[60:210, 30:180] = contour_display

# Properties 
cv2.putText(reference, "CONTOUR PROPERTIES:", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) 
cv2.putText(reference, "cv2.contourArea(c) - Area of contour", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "cv2.arcLength(c, True) - Perimeter", (200, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "cv2.boundingRect(c) - Bounding box", (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "cv2.minEnclosingCircle() - Enclosing circle", (200, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "cv2.moments(c) - Center calculation", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Retrieval modes 
cv2.putText(reference, "RETRIEVAL MODES:", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) 
cv2.putText(reference, "RETR_EXTERNAL - Only outer contours", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "RETR_LIST - All contours, no hierarchy", (200, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "RETR_TREE - Full hierarchy (nested)", (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "RETR_COMP - Two-level hierarchy", (200, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Approximation methods 
cv2.putText(reference, "APPROXIMATION:", (200, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) 
cv2.putText(reference, "CHAIN_APPROX_SIMPLE - Compresses to endpoints", (200, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) 
cv2.putText(reference, "CHAIN_APPROX_NONE - All points", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Shape detection guide 
cv2.putText(reference, "SHAPE DETECTION (by vertices):", (200, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) 
cv2.putText(reference, "3 = Triangle, 4 = Square/Rectangle, 5+ = Polygon/Circle", (200, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# OpenCV 3+ note 
cv2.putText(reference, "OpenCV 3.0+ : image, contours, hierarchy = cv2.findContours()", (30, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 
cv2.putText(reference, "Use [1:] to ignore first return value for compatibility", (30, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 
cv2.putText(reference, "contours, hierarchy = cv2.findContours(...)[1:]", (30, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

cv2.imshow("CONTOUR REFERENCE GUIDE", reference) 
cv2.waitKey(0) 
cv2.imwrite("contour_reference.png", reference) 
print("Saved: contour_reference.png")

print("\n" + "=" * 70) 
print("DAY 13 COMPLETE!") 
print("=" * 70)

cv2.destroyAllWindows()