# exercise_2_document_binarization.py
import cv2
import numpy as np

def binarize_document(image, method="auto"):
    """ Binarize a document image for OCR preprocessing """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Remove noise with blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "simple":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    else:
        # auto - choose based on image characteristics
        hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Check if histogram is bimodal with significant peaks
        peaks = 0
        avg_hist_val = np.mean(hist)
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > avg_hist_val:
                peaks += 1

        if peaks >= 2:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    # Clean up: remove small noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

# Test on different document types
doc_types = ["normal", "shadowed", "low_contrast"]

# Create test documents
normal_doc = np.ones((400, 600), dtype="uint8") * 240
cv2.putText(normal_doc, "Normal Document Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 50, 3)

# Background gradient simulation for shadowed text
shadowed_doc = normal_doc.copy()
shadowed_doc[:, 300:] = 140  # Darker on right
cv2.putText(shadowed_doc, "Shadowed Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 3)

low_contrast_doc = np.ones((400, 600), dtype="uint8") * 150
cv2.putText(low_contrast_doc, "Low Contrast Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 110, 3)

docs = [("Normal", normal_doc), ("Shadowed", shadowed_doc), ("Low Contrast", low_contrast_doc)]

for name, doc in docs:
    binary = binarize_document(doc, "auto")
    
    # Standardize grayscale/binary display channels to match color maps smoothly
    doc_color = cv2.cvtColor(doc, cv2.COLOR_GRAY2BGR)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    comparison = np.hstack([
        cv2.resize(doc_color, (300, 200)),
        cv2.resize(binary_color, (300, 200))
    ])

    cv2.putText(comparison, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(f"Document Binarization - {name}", comparison)
    cv2.waitKey(0)

cv2.destroyAllWindows()