# day9_histograms.py
# Complete guide to image histograms
from __future__ import print_function 
from matplotlib import pyplot as plt 
import numpy as np 
import argparse 
import cv2 
print("=" * 70)
print("DAY 9: HISTOGRAMS")
print("=" * 70)
# -----------------------------------------------------------------
# SECTION 1: LOAD IMAGE AND SETUP
# -----------------------------------------------------------------
print("\n[Section 1] Loading image...")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
# Load image
image_bgr = cv2.imread(args["image"])
if image_bgr is None:
   print("ERROR: Could not load image!")
   exit()
height, width = image_bgr.shape[:2]
print(f"Loaded: {width} x {height} pixels")
# Convert to grayscale for histogram basics
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image", image_bgr)
cv2.imshow("Grayscale", image_gray)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 2: GRAYSCALE HISTOGRAM
# -----------------------------------------------------------------
print("\n[Section 2] Computing Grayscale Histogram...")
print("-" * 50)
# Method 1: Using cv2.calcHist
print("Method 1: cv2.calcHist()")
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
print(f"Histogram shape: {hist.shape}")
print(f"Total pixels counted: {np.sum(hist)} (should equal {height * width})")
# Method 2: Using NumPy (alternative)
print("\nMethod 2: NumPy histogram")
hist_np, bins = np.histogram(image_gray.ravel(), 256, [0, 256])
print(f"NumPy histogram shape: {hist_np.shape}")
# Verify both methods give same result
print(f"Methods match: {np.allclose(hist.flatten(), hist_np)}")
# Visualize using matplotlib
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.plot(hist, color='black')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity (0 = black, 255 = white)")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grayscale_histogram.png")
plt.show()
print("\nInterpretation:")
print("- Peaks on left = dark areas")
print("- Peaks on right = bright areas")
print("- Spread across range = good contrast")
# -----------------------------------------------------------------
# SECTION 3: COLOR HISTOGRAMS
# -----------------------------------------------------------------
print("\n[Section 3] Computing Color Histograms...")
print("-" * 50)
# Split into channels
# Remember: OpenCV is BGR order!
channels = cv2.split(image_bgr)
colors = ("blue", "green", "red")
color_values = ("b", "g", "r")
plt.figure(figsize=(10, 6))
# Plot original image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
# Plot individual channel histograms
plt.subplot(2, 2, 2)
for channel, color, name in zip(channels, color_values, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f"{name.capitalize()}",linewidth=1.5)
plt.title("Color Histograms (Individual Channels)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.legend()
plt.grid(True, alpha=0.3)
# Create color-coded visualizations
plt.subplot(2, 2, 3)
# Blue channel visualized
blue_only = cv2.merge([channels[0], np.zeros_like(channels[0]),
np.zeros_like(channels[0])])
plt.imshow(cv2.cvtColor(blue_only, cv2.COLOR_BGR2RGB))
plt.title("Blue Channel Only")
plt.axis('off')
plt.subplot(2, 2, 4)
# Green channel visualized
green_only = cv2.merge([np.zeros_like(channels[0]), channels[1],
np.zeros_like(channels[0])])
plt.imshow(cv2.cvtColor(green_only, cv2.COLOR_BGR2RGB))
plt.title("Green Channel Only")
plt.axis('off')
plt.tight_layout()
plt.savefig("color_histograms.png")
plt.show()
# Print statistical analysis
print("\n--- Channel Statistics ---")
for channel, name in zip(channels, colors):
    print(f"{name.capitalize()} Channel:")
    print(f" Min: {np.min(channel)}")
    print(f" Max: {np.max(channel)}")
    print(f" Mean: {np.mean(channel):.1f}")
    print(f" Std Dev: {np.std(channel):.1f}")
# -----------------------------------------------------------------
# SECTION 4: 2D HISTOGRAMS (Two Channels at Once)
# -----------------------------------------------------------------
print("\n[Section 4] Computing 2D Histograms...")
print("-" * 50)
print("2D histograms show relationship between TWO channels")
print("Example: How many pixels have Blue=100 AND Green=50?")
# Create 2D histograms for channel pairs
# Using 32 bins for better visualization (32x32 = 1024 combinations)
# Blue vs Green (channels 0 and 1)
hist_bg = cv2.calcHist([channels[0], channels[1]], [0, 1], None, [32,32], [0, 256, 0, 256])
# Green vs Red (channels 1 and 2)
hist_gr = cv2.calcHist([channels[1], channels[2]], [0, 1], None, [32,32], [0, 256, 0, 256])
# Blue vs Red (channels 0 and 2)
hist_br = cv2.calcHist([channels[0], channels[2]], [0, 1], None, [32,32], [0, 256, 0, 256])
print(f"2D Histogram shape: {hist_bg.shape} (32 bins x 32 bins)")
# Visualize 2D histograms
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(hist_bg, interpolation='nearest', cmap='hot')
plt.colorbar(label='Pixel Count')
plt.title("Blue vs Green\n(Blue → X, Green → Y)")
plt.xlabel("Blue Intensity (binned)")
plt.ylabel("Green Intensity (binned)")
plt.subplot(1, 3, 2)
plt.imshow(hist_gr, interpolation='nearest', cmap='hot')
plt.colorbar(label='Pixel Count')
plt.title("Green vs Red\n(Green → X, Red → Y)")
plt.xlabel("Green Intensity (binned)")
plt.ylabel("Red Intensity (binned)")
plt.subplot(1, 3, 3)
plt.imshow(hist_br, interpolation='nearest', cmap='hot')
plt.colorbar(label='Pixel Count')
plt.title("Blue vs Red\n(Blue → X, Red → Y)")
plt.xlabel("Blue Intensity (binned)")
plt.ylabel("Red Intensity (binned)")
plt.tight_layout()
plt.savefig("2d_histograms.png")
plt.show()
print("\nInterpretation:")
print("- Bright spots = common color combinations")
print("- Dark spots = rare color combinations")
# -----------------------------------------------------------------
# SECTION 5: 3D HISTOGRAM (All Three Channels)
# -----------------------------------------------------------------
print("\n[Section 5] 3D Histogram (All Channels Together)...")
print("-" * 50)
# 3D histogram with 8 bins per channel (8x8x8 = 512 combinations)
hist_3d = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256,0, 256, 0, 256])
print(f"3D Histogram shape: {hist_3d.shape}")
print(f"Total bins: {8 * 8 * 8} = 512")
print(f"Non-zero bins: {np.count_nonzero(hist_3d)}")
# Find most common color combination
max_idx = np.unravel_index(np.argmax(hist_3d), hist_3d.shape)
print(f"\nMost common color bin (B,G,R): {max_idx}")
print(f"Bin range: each bin covers 32 intensity values (0-31, 32-63, etc.)")
# Create a visualization of the 3D histogram as a scatter plot
print("\nCreating 3D histogram visualization...")
coords = np.argwhere(hist_3d > 0)
values = hist_3d[hist_3d > 0]
if len(coords) > 0:
 # Take top 500 points for visualization
    top_indices = np.argsort(values)[-500:]
    top_coords = coords[top_indices]
    top_values = values[top_indices]
 
 # Normalize for size
    sizes = (top_values / np.max(top_values) * 50 + 10).flatten()
 
 # Create 2D projection (Blue vs Green, color by Red)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(top_coords[:, 0], top_coords[:, 1], 
    c=top_coords[:, 2], cmap='Reds', 
    s=sizes, alpha=0.6)
    ax.set_xlabel("Blue Bin (0-7)")
    ax.set_ylabel("Green Bin (0-7)")
    ax.set_title("3D Histogram Projection\n(Bubble size = frequency, Color = Red bin)")
    plt.colorbar(scatter, label='Red Bin')
    plt.savefig("3d_histogram_projection.png")
    plt.show()
# -----------------------------------------------------------------
# SECTION 6: HISTOGRAM EQUALIZATION
# -----------------------------------------------------------------
print("\n[Section 6] Histogram Equalization - Improving Contrast")
print("-" * 50)
print("Histogram equalization stretches the pixel distribution")
print("to improve global contrast.")
# Create a low-contrast test image for demonstration
print("\n--- Creating low-contrast test image ---")
low_contrast = np.clip(image_gray * 0.5 + 100, 0, 255).astype("uint8")
cv2.imshow("Low Contrast Image", low_contrast)
cv2.waitKey(0)
# Apply histogram equalization
equalized = cv2.equalizeHist(low_contrast)
cv2.imshow("After Histogram Equalization", equalized)
cv2.waitKey(0)
# Compare histograms
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(low_contrast, cmap='gray')
plt.title("Low Contrast Image")
plt.axis('off')
plt.subplot(2, 2, 2)
hist_low = cv2.calcHist([low_contrast], [0], None, [256], [0, 256])
plt.plot(hist_low, color='black')
plt.title("Low Contrast Histogram (narrow)")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.xlim([0, 256])
plt.subplot(2, 2, 3)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')
plt.subplot(2, 2, 4)
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.plot(hist_eq, color='black')
plt.title("Equalized Histogram (stretched)")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.xlim([0, 256])
plt.tight_layout()
plt.savefig("histogram_equalization_demo.png")
plt.show()
# Apply to real image
print("\n--- Applying to your image ---")
equalized_real = cv2.equalizeHist(image_gray)
# Show comparison
comparison = np.hstack([image_gray, equalized_real])
cv2.imshow("Original vs Equalized (Left: Original, Right: Equalized)",
comparison)
cv2.waitKey(0)
# Save result
cv2.imwrite("equalized_image.png", equalized_real)
print("Saved: equalized_image.png")
print("\nNote: Histogram equalization works best when:")
print(" - The image has poor contrast")
print(" - Foreground and background are both dark or both light")
print(" - Medical and satellite images (can look unnatural on photographs)")
# -----------------------------------------------------------------
# SECTION 7: HISTOGRAMS WITH MASKS
# -----------------------------------------------------------------
print("\n[Section 7] Histograms with Masks (Focus on Regions)")
print("-" * 50)
print("Masks allow us to compute histograms for specific image regions only.")
# Create a mask (circular region in center)
mask = np.zeros(image_gray.shape[:2], dtype="uint8")
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 3
cv2.circle(mask, (center_x, center_y), radius, 255, -1)
cv2.imshow("Circular Mask", mask)
cv2.waitKey(0)
# Apply mask to image
masked_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
cv2.imshow("Masked Region Only", masked_image)
cv2.waitKey(0)
# Compute histogram for masked region
hist_masked = cv2.calcHist([image_gray], [0], mask, [256], [0, 256])
hist_full = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
# Compare histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_full, color='blue', label='Full Image', alpha=0.7)
plt.plot(hist_masked, color='red', label='Masked Region', alpha=0.7)
plt.title("Full Image vs Masked Region")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.legend()
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
# Normalize for better comparison (as percentages)
hist_full_norm = hist_full / np.sum(hist_full) * 100
hist_masked_norm = hist_masked / np.sum(hist_masked) * 100
plt.plot(hist_full_norm, color='blue', label='Full Image', alpha=0.7)
plt.plot(hist_masked_norm, color='red', label='Masked Region', alpha=0.7)
plt.title("Normalized Comparison (%)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Percentage of Pixels")
plt.legend()
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("masked_histogram_comparison.png")
plt.show()
print("\nObservation: Different regions can have very different histograms!")
print("Masked histograms help analyze specific objects in an image.")
# -----------------------------------------------------------------
# SECTION 8: MULTIPLE MASKS - REGION COMPARISON
# -----------------------------------------------------------------
print("\n[Section 8] Comparing Multiple Regions")
print("-" * 50)
# Create three masks: top, center, bottom
height, width = image_gray.shape
mask_top = np.zeros(image_gray.shape, dtype="uint8")
mask_top[0:height//3, :] = 255
mask_center = np.zeros(image_gray.shape, dtype="uint8")
mask_center[height//3:2*height//3, :] = 255
mask_bottom = np.zeros(image_gray.shape, dtype="uint8")
mask_bottom[2*height//3:height, :] = 255
# Compute histograms for each region
hist_top = cv2.calcHist([image_gray], [0], mask_top, [256], [0, 256])
hist_center = cv2.calcHist([image_gray], [0], mask_center, [256], [0,256])
hist_bottom = cv2.calcHist([image_gray], [0], mask_bottom, [256], [0,256])
# Visualize
plt.figure(figsize=(12, 8))
# Show regions
plt.subplot(2, 2, 1)
region_vis = image_bgr.copy()
region_vis[mask_top == 255] = (255, 0, 0)
region_vis[mask_center == 255] = (0, 255, 0)
region_vis[mask_bottom == 255] = (0, 0, 255)
plt.imshow(cv2.cvtColor(region_vis, cv2.COLOR_BGR2RGB))
plt.title("Regions: Top(Blue), Center(Green), Bottom(Red)")
plt.axis('off')
# Plot histograms
plt.subplot(2, 2, 2)
plt.plot(hist_top, color='blue', label='Top Region', linewidth=1.5)
plt.plot(hist_center, color='green', label='Center Region',linewidth=1.5)
plt.plot(hist_bottom, color='red', label='Bottom Region', linewidth=1.5)
plt.title("Histogram by Region")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.legend()
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)
# Normalized comparison
plt.subplot(2, 2, 3)
hist_top_norm = hist_top / np.sum(hist_top) * 100
hist_center_norm = hist_center / np.sum(hist_center) * 100
hist_bottom_norm = hist_bottom / np.sum(hist_bottom) * 100
plt.plot(hist_top_norm, color='blue', label='Top', alpha=0.7)
plt.plot(hist_center_norm, color='green', label='Center', alpha=0.7)
plt.plot(hist_bottom_norm, color='red', label='Bottom', alpha=0.7)
plt.title("Normalized Comparison (%)")
plt.xlabel("Intensity")
plt.ylabel("Percentage")
plt.legend()
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("region_comparison.png")
plt.show()
# -----------------------------------------------------------------
# SECTION 9: PRACTICAL APPLICATION - IMAGE ANALYSIS REPORT
# -----------------------------------------------------------------
print("\n[Section 9] Generating Image Analysis Report")
# Compute comprehensive statistics
hist_full = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
# Find percentiles
cumulative = np.cumsum(hist_full)
total_pixels = cumulative[-1]
def find_percentile(percent):
    target = total_pixels * percent / 100
    return np.searchsorted(cumulative, target)
p5 = find_percentile(5)
p25 = find_percentile(25)
p50 = find_percentile(50)
p75 = find_percentile(75)
p95 = find_percentile(95)
print("\n--- Image Analysis Report ---")
print(f"Image: {args['image']}")
print(f"Dimensions: {width} x {height}")
print(f"\nHistogram Statistics:")
print(f" Darkest 5% of pixels: ≤ {p5}")
print(f" Darkest 25%: ≤ {p25}")
print(f" Median (50%): {p50}")
print(f" Brightest 25%: ≥ {p75}")
print(f" Brightest 5%: ≥ {p95}")
print(f"\nContrast Assessment:")
print(f" Intensity range: {np.min(image_gray)} - {np.max(image_gray)}")
if p95 - p5 < 100:
    print(" → LOW CONTRAST: Consider histogram equalization")
elif p95 - p5 < 150:
    print(" → MEDIUM CONTRAST: Good for most purposes")
else:
    print(" → HIGH CONTRAST: Strong range of intensities")
# Determine if image is dark or bright
if p50 < 85:
   print(" → DARK IMAGE: Predominantly dark tones")
elif p50 > 170:
    print(" → BRIGHT IMAGE: Predominantly light tones")
else:
    print(" → BALANCED: Good mix of dark and light")
# -----------------------------------------------------------------
# SECTION 10: CREATE HISTOGRAM REFERENCE GUIDE
# -----------------------------------------------------------------
print("\n[Section 10] Creating Histogram Reference Guide")
reference = np.zeros((600, 800, 3), dtype="uint8")
ref_height, ref_width = reference.shape[:2]
# Title
cv2.putText(reference, "HISTOGRAM REFERENCE GUIDE", (ref_width//2 - 200,40),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
# Section 1: Original and histogram
orig_small = cv2.resize(image_bgr, (200, 150))
reference[60:210, 20:220] = orig_small 
# Create histogram visualization
hist_vis = np.zeros((150, 200, 3), dtype="uint8")
hist_norm = cv2.normalize(hist_full, None, 0, 150, cv2.NORM_MINMAX)
for i in range(256):
    cv2.line(hist_vis, (i * 200 // 256, 150), 
    (i * 200 // 256, 150 - int(hist_norm[i])), 
    (0, 255, 0), 1)
reference[60:210, 240:440] = hist_vis 
cv2.putText(reference, "Original", (70, 230), cv2.FONT_HERSHEY_SIMPLEX,
0.5, (255, 255, 255), 1)
cv2.putText(reference, "Histogram", (300, 230), cv2.FONT_HERSHEY_SIMPLEX,
0.5, (255, 255, 255), 1)
# Section 2: Color histograms
color_hist_vis = np.zeros((150, 300, 3), dtype="uint8")
for i, (channel, color) in enumerate(zip(channels, color_values)):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    hist_norm = cv2.normalize(hist, None, 0, 100, cv2.NORM_MINMAX)
    for x in range(256):
        cv2.line(color_hist_vis, (x * 300 // 256, 130 - i*30),
        (x * 300 // 256, 130 - i*30 - int(hist_norm[x])),
        (0, 255, 0) if color == 'g' else 
        (255, 0, 0) if color == 'b' else 
        (0, 0, 255), 1)
reference[230:380, 20:320] = color_hist_vis 
cv2.putText(reference, "B", (40, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 1)
cv2.putText(reference, "G", (150, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
cv2.putText(reference, "R", (260, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
# Section 3: Equalization demo
eq_small = cv2.resize(equalized_real, (150, 150))
reference[230:380, 450:600] = cv2.cvtColor(eq_small, cv2.COLOR_GRAY2BGR)
cv2.putText(reference, "Equalized", (490, 400), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
# Footer notes
cv2.putText(reference, "KEY INSIGHTS:", (20, 440),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.putText(reference, "1. Histograms show pixel intensity distribution",(20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "2. Narrow histogram = low contrast", (20, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "3. Use cv2.calcHist() for analysis", (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "4. Masks focus analysis on specific regions",(20, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "5. Histogram equalization improves contrast",(20, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.imshow("HISTOGRAM REFERENCE GUIDE", reference)
cv2.waitKey(0)
cv2.imwrite("histogram_reference.png", reference)
print("Saved: histogram_reference.png")
print("\n" + "=" * 70)
print("DAY 9 COMPLETE!")
print("=" * 70)
cv2.destroyAllWindows()
plt.close('all')