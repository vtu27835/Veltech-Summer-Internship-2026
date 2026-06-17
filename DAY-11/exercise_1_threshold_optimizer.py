#exercise_1_threshold_optimizer.py
import cv2
import numpy as np

def threshold_optimizer(image_path):
    """Find optimal threshold by analyzing histogram"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}!")
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Find distinct peaks in histogram (filtering out minor noise)
    peaks = []
    avg_hist_val = np.mean(hist)
    
    for i in range(1, 255):
        # Must be higher than neighbors AND higher than average to count as a major peak
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > avg_hist_val:
            peaks.append((i, hist[i]))

    # Sort peaks by height (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)

    if len(peaks) >= 2:
        # Optimal threshold is between two highest peaks
        valley_start = min(peaks[0][0], peaks[1][0])
        valley_end = max(peaks[0][0], peaks[1][0])

        # Find minimum between peaks
        min_val = float('inf')
        optimal_T = (valley_start + valley_end) // 2

        for i in range(valley_start, valley_end + 1):
            if hist[i] < min_val:
                min_val = hist[i]
                optimal_T = i

        print(f"Detected two major peaks at {peaks[0][0]} and {peaks[1][0]}")
        print(f"Optimal threshold T = {optimal_T}")
    else:
        optimal_T = 127
        print(f"Only {len(peaks)} prominent peak(s) found. Using default T=127")

    # Apply threshold
    _, thresholded = cv2.threshold(gray, optimal_T, 255, cv2.THRESH_BINARY)

    # Convert grayscale images to 3 channels for a cleaner side-by-side color canvas display
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thresholded_color = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

    # Show results
    comparison = np.hstack([
        cv2.resize(gray_color, (300, 250)),
        cv2.resize(thresholded_color, (300, 250))
    ])

    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison, f"Optimal T = {optimal_T}", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Threshold Optimizer", comparison)
    cv2.waitKey(0)

    return optimal_T

# Run optimizer
threshold_optimizer("test_image.png")
cv2.destroyAllWindows()