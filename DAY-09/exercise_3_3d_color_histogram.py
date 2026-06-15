# exercise_3_3d_color_histogram.py
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
image = cv2.imread("test_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Sample pixels for visualization (use subset for performance)
h, w = image.shape[:2]
sample = image_rgb.reshape(-1, 3)
sample = sample[np.random.choice(len(sample), min(5000, len(sample)),replace=False)]
# Create 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
# Plot pixels in RGB space
ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=sample / 255, s=5, alpha=0.5)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('3D RGB Color Space Distribution')
# Add axis limits
ax.set_xlim([0, 255])
ax.set_ylim([0, 255])
ax.set_zlim([0, 255])
plt.savefig("3d_color_distribution.png")
plt.show()
# Create 2D projections
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
projections = [
 (0, 1, "Red vs Green", "Red", "Green"),
 (0, 2, "Red vs Blue", "Red", "Blue"),
 (1, 2, "Green vs Blue", "Green", "Blue")
]
for idx, (x_idx, y_idx, title, xlabel, ylabel) in enumerate(projections):
    axes[idx].scatter(sample[:, x_idx], sample[:, y_idx], 
    c=sample / 255, s=3, alpha=0.5)
    axes[idx].set_xlabel(xlabel)
    axes[idx].set_ylabel(ylabel)
    axes[idx].set_title(title)
    axes[idx].set_xlim([0, 255])
    axes[idx].set_ylim([0, 255])
plt.tight_layout()
plt.savefig("2d_color_projections.png")
plt.show()