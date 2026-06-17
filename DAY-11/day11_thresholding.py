# day11_thresholding.py # Complete guide to image thresholding
from __future__ import print_function
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

print("=" * 70)
print("DAY 11: THRESHOLDING")
print("=" * 70)

# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE
# -----------------------------------------------------------------
print("\n[Section 1] Loading image...")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Load image
image = cv2.imread(args["image"])
if image is None:
    print("ERROR: Could not load image!")
    exit()

height, width = image.shape[:2]
print(f"Loaded: {width} x {height} pixels")

# Convert to grayscale (thresholding works on single channel)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original (Color)", image)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# Create a test image with varying lighting for adaptive thresholding demo
print("\nCreating test image with varying lighting...")
test_image = np.zeros((300, 500), dtype="uint8")
# Left side: dark text on light background
cv2.rectangle(test_image, (0, 0), (250, 300), 200, -1)
cv2.putText(test_image, "DARK", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 50, 3)
# Right side: light text on dark background (inverse)
cv2.rectangle(test_image, (250, 0), (500, 300), 50, -1)
cv2.putText(test_image, "LIGHT", (330, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 3)

# Add gradient in middle for smooth transition
for x in range(250, 500):
    intensity = int(50 + (x - 250) * 150 / 250)
    test_image[:, x] = intensity

cv2.imshow("Test Image (Varying Lighting)", test_image)
cv2.waitKey(0)

# -----------------------------------------------------------------
# SECTION 2: SIMPLE THRESHOLDING
# -----------------------------------------------------------------
print("\n[Section 2] Simple Thresholding")
print("-" * 50)
print("Simple thresholding uses a SINGLE global threshold value T.")
print("Formula: pixel = 255 if pixel > T else 0")

# Try different threshold values
thresholds = [50, 100, 127, 150, 200]
print("\nApplying different threshold values:")

for T in thresholds:
    # Apply binary threshold
    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    # Apply inverse binary threshold
    _, binary_inv = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY_INV)
    
    # Display
    cv2.putText(binary, f"Binary (T={T})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(binary_inv, f"Binary Inv (T={T})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    comparison = np.hstack([binary, binary_inv])
    cv2.imshow(f"Simple Threshold - T={T}", comparison)
    cv2.waitKey(0)

print("\nObservation: Lower T = more white pixels, Higher T = more black pixels")
print("Finding the 'perfect' T requires trial and error!")

# Test on varying lighting image
print("\n--- Simple Threshold on Varying Lighting (Problem!) ---")
for T in [100, 127, 150]:
    _, binary_test = cv2.threshold(test_image, T, 255, cv2.THRESH_BINARY)
    cv2.putText(binary_test, f"T={T} (fails on one side)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(f"Simple Threshold on Varying Lighting - T={T}", binary_test)
    cv2.waitKey(0)

print("PROBLEM: Single threshold can't handle both dark and light regions!")
print("Solution: Adaptive thresholding!")

# -----------------------------------------------------------------
# SECTION 3: ADAPTIVE THRESHOLDING
# -----------------------------------------------------------------
print("\n[Section 3] Adaptive Thresholding")
print("-" * 50)
print("Adaptive thresholding uses DIFFERENT thresholds for different regions.")
print("How it works: For each pixel, look at its neighborhood and compute:")
print("\nTwo methods:")
print(" 1. MEAN_C: Threshold = mean(neighborhood) - C")
print(" 2. GAUSSIAN_C: Threshold = weighted mean(neighborhood) - C")

# Apply adaptive thresholding
block_sizes = [11, 21, 31]  # Neighborhood size (must be odd)
constant_C = [2, 5, 10]     # Value subtracted from mean

print("\nApplying adaptive thresholding to varying lighting image...")
# Show original test image
cv2.imshow("Original Test Image (varying lighting)", test_image)
cv2.waitKey(0)

# Method 1: Mean C
print("\n--- Method: ADAPTIVE_THRESH_MEAN_C ---")
for block in block_sizes:
    for C in constant_C:
        adaptive_mean = cv2.adaptiveThreshold(
            test_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C
        )
        cv2.putText(adaptive_mean, f"Mean C: block={block}, C={C}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(f"Adaptive Mean - block={block}, C={C}", adaptive_mean)
        cv2.waitKey(0)

# Method 2: Gaussian C
print("\n--- Method: ADAPTIVE_THRESH_GAUSSIAN_C ---")
for block in block_sizes:
    for C in constant_C:
        adaptive_gauss = cv2.adaptiveThreshold(
            test_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C
        )
        cv2.putText(adaptive_gauss, f"Gauss C: block={block}, C={C}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(f"Adaptive Gaussian - block={block}, C={C}", adaptive_gauss)
        cv2.waitKey(0)

print("\nSUCCESS! Adaptive thresholding handles varying lighting!")
print(" - Mean C: faster, good for uniform lighting variations")
print(" - Gaussian C: slower, better for complex lighting")

# Apply to real image
print("\n--- Applying Adaptive Thresholding to Your Image ---")
# Find good parameters (tune these for your image)
best_adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
cv2.imshow("Original Grayscale", gray)
cv2.imshow("Adaptive Threshold (Gaussian, block=11, C=2)", best_adaptive)
cv2.waitKey(0)

# -----------------------------------------------------------------
# SECTION 4: OTSU'S METHOD (AUTOMATIC THRESHOLD)
# -----------------------------------------------------------------
print("\n[Section 4] Otsu's Method - Automatic Threshold Selection")
print("-" * 50)
print("Otsu's method analyzes the histogram to find the optimal threshold.")
print("Assumption: Image has two peaks (foreground and background).")

# Compute and display histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Find Otsu's threshold
otsu_threshold, otsu_result = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"\nOtsu automatically found threshold T = {otsu_threshold}")

# Compare with manual threshold
manual_threshold = 127
_, manual_result = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)

# Create comparison
comparison = np.hstack([
    cv2.resize(gray, (250, 200)),
    cv2.resize(manual_result, (250, 200)),
    cv2.resize(otsu_result, (250, 200))
])

cv2.putText(comparison, "Original", (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(comparison, f"Manual T={manual_threshold}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(comparison, f"Otsu T={otsu_threshold}", (560, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Manual vs Otsu Thresholding", comparison)
cv2.waitKey(0)

# Visualize threshold on histogram
print("\n--- Visualizing Otsu Threshold on Histogram ---")
plt.figure(figsize=(10, 6))
plt.plot(hist, color='black')
plt.axvline(x=otsu_threshold, color='red', linestyle='--', label=f"Otsu Threshold = {otsu_threshold}")
plt.axvline(x=manual_threshold, color='blue', linestyle='--', label=f"Manual Threshold = {manual_threshold}")
plt.title("Grayscale Histogram with Thresholds")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("otsu_histogram.png")
plt.show()

print("\nAdvantages of Otsu:")
print(" - No manual tuning needed")
print(" - Optimal for bimodal histograms (two distinct peaks)")
print("Disadvantages:")
print(" - Fails if histogram is not bimodal")
print(" - Assumes uniform lighting")

# -----------------------------------------------------------------
# SECTION 5: THRESHOLDING TYPES COMPARISON
# -----------------------------------------------------------------
print("\n[Section 5] All Thresholding Types Comparison")
print("-" * 50)

# Apply all types for comparison
threshold_types = {
    "Binary (T=127)": lambda: cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
    "Binary Inv (T=127)": lambda: cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1],
    "Trunc (T=127)": lambda: cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)[1],
    "Tozero (T=127)": lambda: cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)[1],
    "Tozero Inv (T=127)": lambda: cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)[1],
    "Adaptive (Mean)": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "Adaptive (Gaussian)": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
    "Otsu": lambda: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
}

# Create a grid of results
grid_images = []
current_row = []
for i, (name, func) in enumerate(threshold_types.items()):
    result = func()
    result_resized = cv2.resize(result, (200, 150))
    cv2.putText(result_resized, name[:15], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    current_row.append(result_resized)
    
    if len(current_row) == 4:
        grid_images.append(np.hstack(current_row))
        current_row = []

if current_row:
    while len(current_row) < 4:
        blank = np.zeros((150, 200), dtype="uint8")
        current_row.append(blank)
    grid_images.append(np.hstack(current_row))

grid = np.vstack(grid_images)
cv2.imshow("ALL THRESHOLDING TYPES COMPARED", grid)
cv2.waitKey(0)

# -----------------------------------------------------------------
# SECTION 6: PRACTICAL APPLICATION - COINS SEGMENTATION
# -----------------------------------------------------------------
print("\n[Section 6] Practical Application: Coin Segmentation")
print("-" * 50)

# Create a simulated coins image
coins = np.zeros((400, 400), dtype="uint8")

# Draw circles to simulate coins
cv2.circle(coins, (100, 100), 50, 200, -1)
cv2.circle(coins, (250, 120), 45, 180, -1)
cv2.circle(coins, (180, 250), 55, 220, -1)
cv2.circle(coins, (320, 280), 40, 190, -1)
cv2.circle(coins, (80, 300), 35, 170, -1)

# Add some noise
noise = np.random.randint(0, 30, coins.shape, dtype="uint8")
coins_noisy = cv2.add(coins, noise)

cv2.imshow("Coins Image (simulated)", coins_noisy)
cv2.waitKey(0)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(coins_noisy, (5, 5), 0)

# Apply Otsu's thresholding
_, coin_threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours (preview of Day 13)
contours, _ = cv2.findContours(coin_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count coins
coin_count = len(contours)
print(f"Found {coin_count} coins!")

# Draw contours on original
result = cv2.cvtColor(coins_noisy, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Show results
comparison = np.hstack([
    cv2.resize(coins_noisy, (250, 250)),
    cv2.resize(coin_threshold, (250, 250)),
    cv2.resize(result, (250, 250))
])

cv2.putText(comparison, "Original", (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(comparison, "Threshold (Otsu)", (300, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(comparison, f"Detected: {coin_count} coins", (540, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Coin Segmentation Pipeline", comparison)
cv2.waitKey(0)

print("\nSegmentation Pipeline Steps:")
print(" 1. Convert to grayscale ✓")
print(" 2. Apply blur to reduce noise ✓")
print(" 3. Apply Otsu threshold ✓")
print(" 4. Find contours to count objects ✓")

# -----------------------------------------------------------------
# SECTION 7: PRACTICAL APPLICATION - DOCUMENT SCANNING
# -----------------------------------------------------------------
print("\n[Section 7] Practical Application: Document Scanning Prep")
print("-" * 50)

# Create a document simulation
document = np.ones((500, 500), dtype="uint8") * 200  # Light gray background

# Add text
cv2.putText(document, "IMPORTANT DOCUMENT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 3)
cv2.putText(document, "This is a sample document", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 80, 2)
cv2.putText(document, "with dark text on light background.", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 80, 2)

# Add a dark box
cv2.rectangle(document, (100, 250), (400, 350), 100, -1)
cv2.putText(document, "DARK SECTION", (150, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)

cv2.imshow("Document (with dark and light sections)", document)
cv2.waitKey(0)

# Different thresholding methods for document
_, binary_doc = cv2.threshold(document, 127, 255, cv2.THRESH_BINARY)
adaptive_doc = cv2.adaptiveThreshold(document, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

comparison_doc = np.hstack([
    cv2.resize(document, (250, 250)),
    cv2.resize(binary_doc, (250, 250)),
    cv2.resize(adaptive_doc, (250, 250))
])

cv2.putText(comparison_doc, "Original", (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(comparison_doc, "Simple (T=127)", (290, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(comparison_doc, "Adaptive", (550, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Document Thresholding Comparison", comparison_doc)
cv2.waitKey(0)

print("Conclusion: Adaptive thresholding handles mixed lighting/document sections better!")

# -----------------------------------------------------------------
# SECTION 8: INTERACTIVE THRESHOLD EXPLORER
# -----------------------------------------------------------------
print("\n[Section 8] Interactive Threshold Explorer")
print("-" * 50)

def interactive_threshold(image_gray):
    """Interactive tool to explore thresholding with trackbars"""
    cv2.namedWindow("Threshold Explorer")
    
    # Create trackbars
    cv2.createTrackbar("Threshold", "Threshold Explorer", 127, 255, lambda x: None)
    cv2.createTrackbar("Method", "Threshold Explorer", 0, 3, lambda x: None)
    
    methods = ["Binary", "Binary Inv", "Trunc", "Tozero"]
    
    print("\nInteractive Threshold Explorer:")
    print(" - Use trackbars to adjust threshold and method")
    print(" - Press ESC to exit")
    
    while True:
        T = cv2.getTrackbarPos("Threshold", "Threshold Explorer")
        method_idx = cv2.getTrackbarPos("Method", "Threshold Explorer")
        
        method_map = {
            0: cv2.THRESH_BINARY,
            1: cv2.THRESH_BINARY_INV,
            2: cv2.THRESH_TRUNC,
            3: cv2.THRESH_TOZERO
        }
        
        _, result = cv2.threshold(image_gray, T, 255, method_map[method_idx])
        
        # Convert to color for display
        result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        original_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        
        # Add info
        cv2.putText(result_color, f"Method: {methods[method_idx]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_color, f"Threshold: {T}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        display = np.hstack([original_color, result_color])
        cv2.imshow("Threshold Explorer", display)
        
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break
            
    cv2.destroyAllWindows()

# Run interactive explorer
interactive_threshold(gray)

# -----------------------------------------------------------------
# SECTION 9: CREATE THRESHOLDING REFERENCE GUIDE
# -----------------------------------------------------------------
print("\n[Section 9] Creating Thresholding Reference Guide")

# Increased canvas height to 800 so everything fits without generating a ValueError
reference = np.zeros((800, 800, 3), dtype="uint8")
ref_height, ref_width = reference.shape[:2]

# Title
cv2.putText(reference, "THRESHOLDING REFERENCE GUIDE", (ref_width//2 - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Create test pattern for demonstration
test_pattern = np.zeros((100, 800), dtype="uint8")
for x in range(800):
    intensity = int(x * 255 / 800)
    test_pattern[:, x] = intensity

# Apply different thresholds
_, binary_pattern = cv2.threshold(test_pattern, 127, 255, cv2.THRESH_BINARY)
_, binary_inv_pattern = cv2.threshold(test_pattern, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc_pattern = cv2.threshold(test_pattern, 127, 255, cv2.THRESH_TRUNC)
_, tozero_pattern = cv2.threshold(test_pattern, 127, 255, cv2.THRESH_TOZERO)

# Place in reference
reference[60:160, 0:800] = cv2.cvtColor(test_pattern, cv2.COLOR_GRAY2BGR)
reference[180:280, 0:800] = cv2.cvtColor(binary_pattern, cv2.COLOR_GRAY2BGR)
reference[300:400, 0:800] = cv2.cvtColor(binary_inv_pattern, cv2.COLOR_GRAY2BGR)
reference[420:520, 0:800] = cv2.cvtColor(trunc_pattern, cv2.COLOR_GRAY2BGR)
reference[540:640, 0:800] = cv2.cvtColor(tozero_pattern, cv2.COLOR_GRAY2BGR)

# Labels
cv2.putText(reference, "Original Gradient", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(reference, "BINARY (pixel > T = 255)", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(reference, "BINARY_INV (pixel > T = 0)", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(reference, "TRUNC (cap at T)", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(reference, "TOZERO (pixel < T = 0)", (10, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Add T=127 annotation
cv2.line(reference, (int(800 * 127 / 256), 60), (int(800 * 127 / 256), 160), (0, 0, 255), 2)
cv2.putText(reference, "T=127", (int(800 * 127 / 256) - 20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Formula boxes
formula_y = 660
cv2.putText(reference, "FORMULAS:", (20, formula_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.putText(reference, "BINARY: dst = 255 if src > T else 0", (20, formula_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
cv2.putText(reference, "BINARY_INV: dst = 0 if src > T else 255", (20, formula_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
cv2.putText(reference, "TRUNC: dst = T if src > T else src", (20, formula_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
cv2.putText(reference, "TOZERO: dst = src if src > T else 0", (20, formula_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

cv2.imshow("THRESHOLDING REFERENCE GUIDE", reference)
cv2.waitKey(0)
cv2.imwrite("thresholding_reference.png", reference)
print("Saved: thresholding_reference.png")

print("\n" + "=" * 70)
print("DAY 11 COMPLETE!")
print("=" * 70)

cv2.destroyAllWindows()