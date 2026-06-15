# exercise_5_contrast_enhancement.py
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
def enhance_contrast(image, method='auto'):
    """ 
    Apply contrast enhancement using various methods 
 
    Methods: 
    - 'auto': Automatic selection based on histogram analysis 
    - 'equalize': Histogram equalization 
    - 'clahe': Contrast Limited Adaptive Histogram Equalization 
    - 'linear': Linear contrast stretch 
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    if method == 'equalize':
        enhanced = cv2.equalizeHist(gray)
 
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
 
    elif method == 'linear':
 # Linear contrast stretch
        min_val = np.percentile(gray, 5)
        max_val = np.percentile(gray, 95)
        enhanced = np.clip((gray - min_val) * 255.0 / (max_val -min_val), 0, 255).astype('uint8')
 
    else: # 'auto' - choose best based on histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
 # Check if histogram is already well-distributed
        spread = np.percentile(gray, 95) - np.percentile(gray, 5)
        if spread > 150:
            enhanced = gray # Already good
        elif spread > 80:
            enhanced = cv2.equalizeHist(gray)
        else:
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
 
    return enhanced 
# Load image
image = cv2.imread("test_image.png")
# Apply all methods
methods = ['auto', 'equalize', 'clahe', 'linear']
results = {}
for method in methods:
    enhanced = enhance_contrast(image, method)
 # Convert back to BGR for display
    if len(enhanced.shape) == 2:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    results[method] = enhanced 
# Display comparison
plt.figure(figsize=(15, 10))
for idx, (method, result) in enumerate(results.items()):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Method: {method.upper()}")
    plt.axis('off')
 
 # Add histogram in corner
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_result], [0], None, [256], [0, 256])
 
 # Create inset axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
    axins = inset_axes(plt.gca(), width="40%", height="30%", loc='lower right')
    axins.plot(hist, color='black')
    axins.set_xlim([0, 256])
    axins.set_xticks([0, 128, 256])
    axins.set_title('Histogram', fontsize=8)
plt.tight_layout()
plt.savefig("contrast_enhancement_comparison.png")
plt.show()
# Save best result
best = results['auto']
cv2.imwrite("enhanced_image.png", best)
print("Saved: enhanced_image.png")