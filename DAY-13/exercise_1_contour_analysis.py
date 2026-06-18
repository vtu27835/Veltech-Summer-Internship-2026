import cv2
import numpy as np

def analyze_contours(image_path):
    """Analyze contours and print statistics"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, [], []
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # FIX 1: Modern OpenCV returns (contours, hierarchy). No slicing needed.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Contour Analysis Report")
    print("-" * 50)
    
    areas = []
    perimeters = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        areas.append(area)
        perimeters.append(perimeter)
        # FIX 2: Moved inside the loop to print details for EVERY contour
        print(f"Contour {i+1}: Area={area:.0f}, Perimeter={perimeter:.1f}")
        
    print("-" * 50)
    print(f"Total contours: {len(contours)}")
    
    # FIX 3: Safety check to avoid crashing if no contours are found
    if contours:
        print(f"Total area: {sum(areas):.0f} pixels")
        print(f"Average area: {np.mean(areas):.0f}")
        print(f"Largest area: {max(areas):.0f}")
        print(f"Smallest area: {min(areas):.0f}")
    else:
        print("No contours found in the image.")
        
    return contours, areas, perimeters

# Run analysis
analyze_contours("test_image.png")