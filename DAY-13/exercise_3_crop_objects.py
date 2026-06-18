import cv2
import numpy as np
import os

def crop_and_save_objects(image_path, output_dir="cropped_objects"):
    """Crop each object and save as separate image"""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # FIX 1: Correct unpack for modern OpenCV
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Cropping {len(contours)} objects...")
    
    # FIX 2: Properly indented loop body so every object is processed
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add small padding
        pad = 5
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(image.shape[1], x + w + pad)
        y_end = min(image.shape[0], y + h + pad)
        
        # Crop object using NumPy slicing [rows, columns]
        obj = image[y_start:y_end, x_start:x_end]
        
        # Save to disk
        filename = f"{output_dir}/object_{i+1}.png"
        cv2.imwrite(filename, obj)
        print(f"Saved: {filename}")
        
    print(f"All objects saved to {output_dir}/")

# Run cropping
crop_and_save_objects("test_image.png")
cv2.destroyAllWindows()