import cv2
import numpy as np

def order_points(pts):
    """Sort coordinates in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left point will have the smallest sum, 
    # Bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point will have the smallest difference,
    # Bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def scan_document(image_path):
    """Find and extract document from image using contours"""
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
        return
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find document boundaries
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # FIX 1: Correct unpack for modern OpenCV
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found!")
        return
        
    # Find largest contour (should be the document)
    largest = max(contours, key=cv2.contourArea)
    
    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    print(f"Document corners detected: {len(approx)}")
    
    # Draw preview on original image copy
    result = image.copy()
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)
    
    # Draw corner points
    for point in approx:
        x, y = point[0]
        cv2.circle(result, (x, y), 8, (0, 0, 255), -1)
        
    cv2.imshow("Document Detected", result)
    
    # Crop document (if quadrilateral)
    if len(approx) == 4:
        # Reshape to a clean (4, 2) array
        pts = approx.reshape(4, 2)
        
        # FIX 2: Explicitly order the points to match the target layout
        ordered_pts = order_points(pts)
        (tl, tr, br, bl) = ordered_pts
        
        # Calculate maximum clean width and height using Euclidean distance
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        
        # Define target destination mapping 
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Warp the perspective
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        cv2.imshow("Scanned Document", warped)
        cv2.imwrite("scanned_document.png", warped)
        print("Scanned document saved successfully as scanned_document.png")
    else:
        print("Could not isolate a clean 4-cornered document outline. Check threshold/edges.")
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run document scanner
scan_document("test_image.png")