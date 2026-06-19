# exercise_4_advanced_shape_detection.py
import cv2
import numpy as np

def advanced_shape_detection(image_path):
    """Detect shapes with advanced filtering and logical confidence mapping"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # FIX: Native unpack syntax for modern OpenCV 4+ compatibility
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    shape_data = {}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        vertices = len(approx)
        
        # Calculate circularity metric
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        
        # FIX: Restructured hierarchical logic tree to resolve the circle classification paradox
        if circularity > 0.85:
            shape = "Circle"
            confidence = circularity
        elif vertices == 3:
            shape = "Triangle"
            confidence = 0.90
        elif vertices == 4:
            aspect_ratio = w / float(h) if h > 0 else 1
            shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
            confidence = 0.95 if shape == "Square" else 0.90
        elif vertices == 5:
            shape = "Pentagon"
            confidence = 0.85
        elif vertices == 6:
            shape = "Hexagon"
            confidence = 0.85
        elif 6 < vertices <= 10:
            shape = f"Polygon ({vertices}-sided)"
            confidence = 0.70
        else:
            shape = "Complex/Irregular"
            confidence = 0.40
            
        # Store metadata
        if shape not in shape_data:
            shape_data[shape] = {'count': 0, 'areas': []}
        shape_data[shape]['count'] += 1
        shape_data[shape]['areas'].append(area)
        
        # Rendering
        color = np.random.randint(0, 256, 3).tolist()
        cv2.drawContours(result, [contour], -1, color, 2)
        
        # Centroid text calculations via image moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label = f"{shape} ({confidence:.2f})"
            cv2.putText(result, label, (cx-25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
    # Output analytical summary results
    print("\n" + "=" * 55)
    print("Advanced Shape Detection Summary Results")
    print("-" * 55)
    for shape, data in shape_data.items():
        avg_area = np.mean(data['areas']) if data['areas'] else 0
        print(f" {shape:<20}: {data['count']:>2} instances detected | Avg Area: {avg_area:.0f}px")
    print("=" * 55 + "\n")
    
    # Save visual artifact output
    cv2.imwrite("advanced_shapes.png", result)
    print(" → Processed map layout saved as: 'advanced_shapes.png'")

# -----------------------------------------------------------------
# Trigger Routine Execution
# -----------------------------------------------------------------
if __name__ == "__main__":
    advanced_shape_detection("test_image.png")