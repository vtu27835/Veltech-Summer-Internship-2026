# exercise_2_equalization_grid.py
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
image = cv2.imread("test_image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply different equalization methods
methods = {
 "Original": gray,
 "Standard Equalization": cv2.equalizeHist(gray),
 "CLAHE (Contrast Limiting)": cv2.createCLAHE(clipLimit=2.0,
tileGridSize=(8,8)).apply(gray),
 "Adaptive Histogram": cv2.createCLAHE(clipLimit=3.0,
tileGridSize=(16,16)).apply(gray)
}
# Create comparison grid
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
for idx, (name, img) in enumerate(methods.items()):
    row = idx // 2
    col = (idx % 2) * 2
 
 # Show image
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(name)
    axes[row, col].axis('off')
 
 # Show histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    axes[row, col + 1].plot(hist, color='black')
    axes[row, col + 1].set_title(f"{name} Histogram")
    axes[row, col + 1].set_xlim([0, 256])
    axes[row, col + 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("equalization_comparison.png")
plt.show()