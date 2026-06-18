import cv2
import numpy as np

def classify_shapes(image_path):
    """Classify shapes using contour analysis"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # FIX 1: Correct unpack for modern OpenCV
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    shape_counts = {}
    
    # FIX 2: Properly indented loop body so every shape is evaluated
    for contour in contours:
        # Skip noisy, small artifacts
        if cv2.contourArea(contour) < 50:
            continue
            
        # Approximate contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)  # 0.04 is often a safer baseline
        vertices = len(approx)
        
        # Determine shape based on vertex count
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 1
            # A square has a roughly equal width and height
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices == 6:
            shape = "Hexagon"
        elif vertices > 6:
            shape = "Circle"
        else:
            shape = "Unknown"
            
        # Count shapes
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        # Draw green border around the shape
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # Calculate image moments to find the center (centroid) of the shape
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Label the shape at its center
            cv2.putText(result, shape, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    # Print summary statistics outside the loop
    print("Shape Classification Results:")
    print("-" * 40)
    for shape, count in shape_counts.items():
        print(f"{shape}: {count}")
        
    cv2.imshow("Classified Shapes", result)
    cv2.waitKey(0)
    cv2.imwrite("classified_shapes.png", result)

# Run shape classification
classify_shapes("test_image.png")
cv2.destroyAllWindows()