# exercise_3_multi_object_segmentation.py
import cv2
import numpy as np

def segment_objects(image, min_area=100):
    """ Segment objects from background using thresholding """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up minor noise artifacts from background before tracking contours
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Draw results
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
    cv2.drawContours(result, valid_contours, -1, (0, 255, 0), 2)

    # Add labels
    for i, contour in enumerate(valid_contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, str(i+1), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return result, len(valid_contours), binary

# Create test image with multiple objects
test_objects = np.zeros((400, 600, 3), dtype="uint8")

# Draw different shapes as "objects"
cv2.circle(test_objects, (100, 100), 40, (200, 200, 200), -1)
cv2.circle(test_objects, (250, 120), 30, (180, 180, 180), -1)
cv2.rectangle(test_objects, (50, 250), (120, 350), (190, 190, 190), -1)
cv2.rectangle(test_objects, (200, 280), (300, 360), (170, 170, 170), -1)
cv2.ellipse(test_objects, (450, 150), (50, 30), 0, 0, 360, (200, 200, 200), -1)
cv2.ellipse(test_objects, (480, 300), (40, 60), 0, 0, 360, (180, 180, 180), -1)

# Add some noise
noise = np.random.randint(0, 30, test_objects.shape, dtype="uint8")
test_objects = cv2.add(test_objects, noise)

cv2.imshow("Original Objects", test_objects)
cv2.waitKey(0)

# Segment
result, count, binary = segment_objects(test_objects, min_area=200)

cv2.imshow("Segmented Objects (Binary Mask)", binary)
cv2.imshow(f"Detected {count} Objects", result)
cv2.waitKey(0)

cv2.destroyAllWindows()