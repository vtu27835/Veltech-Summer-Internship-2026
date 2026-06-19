# exercise_2_object_counting.py
import cv2
import numpy as np

def count_objects(image_path, methods=['threshold', 'canny', 'adaptive']):
    """Count objects using different methods and compare performance"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    results = {}
    
    # Method 1: Global Otsu Thresholding
    if 'threshold' in methods:
        # We invert the mask using THRESH_BINARY_INV to guarantee objects are white (255)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # FIX: Removed the legacy trailing slice [1:]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results['threshold'] = len([c for c in contours if cv2.contourArea(c) > 50])
        
    # Method 2: Canny Edges + Morphological Dilate
    if 'canny' in methods:
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        # Dilate closes small gaps in the edge borders before extracting contours
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # FIX: Removed the legacy trailing slice [1:]
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results['canny'] = len([c for c in contours if cv2.contourArea(c) > 50])
        
    # Method 3: Local Adaptive Gaussian Thresholding
    if 'adaptive' in methods:
        # FIX: Changed to THRESH_BINARY_INV so objects are segmented as white masks
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # FIX: Removed the legacy trailing slice [1:]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results['adaptive'] = len([c for c in contours if cv2.contourArea(c) > 50])
        
    # Print clean summary comparison
    print("\n" + "=" * 40)
    print("Object Counting Methods Comparison")
    print("-" * 40)
    for method, count in results.items():
        print(f" {method.capitalize():<12}: {count} objects detected")
    print("=" * 40 + "\n")
    
    return results

# -----------------------------------------------------------------
# Execution Entry Point
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Ensure you replace this with a valid filename for testing
    results = count_objects("test_image.png", ['threshold', 'canny', 'adaptive'])