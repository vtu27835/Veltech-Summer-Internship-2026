# detect_faces.py
# Face detection in static images
from __future__ import print_function
import argparse
import cv2
import os
import numpy as np
from face_detector import FaceDetector
print("=" * 70)
print("DAY 15: FACE DETECTION WITH HAAR CASCADES")
print("=" * 70)
# -----------------------------------------------------------------
# SECTION 1: SETUP AND LOAD IMAGE
# -----------------------------------------------------------------
print("\n[Section 1] Setting up face detection...")
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
 help="path to the face cascade XML file")
ap.add_argument("-i", "--image", required=True,
 help="path to the image file")
ap.add_argument("-s", "--scale", type=float, default=1.1,
 help="scale factor for detection (default: 1.1)")
ap.add_argument("-n", "--neighbors", type=int, default=5,
 help="minimum neighbors for face confirmation (default: 5)")
args = vars(ap.parse_args())
# Load the image
image = cv2.imread(args["image"])
if image is None:
 print("ERROR: Could not load image!")
 exit()
height, width = image.shape[:2]
print(f"Image loaded: {width} x {height} pixels")
# Initialize the face detector
detector = FaceDetector(args["face"])
cv2.imshow("Original Image", image)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 2: PERFORM FACE DETECTION
# -----------------------------------------------------------------
print("\n[Section 2] Detecting faces...")
print(f"Parameters: scaleFactor={args['scale']}, minNeighbors={args['neighbors']}")
# Detect faces
faces = detector.detect(
 image,
 scaleFactor=args['scale'],
 minNeighbors=args['neighbors'],
 minSize=(30, 30)
)
print(f"Found {len(faces)} face(s)")
# Draw boxes on the image
result = detector.draw_boxes(image, faces, color=(0, 255, 0),
thickness=2)
# Add text showing number of faces
cv2.putText(result, f"Faces Detected: {len(faces)}", (10, 30),
 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Face Detection Results", result)
cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 3: DISPLAY INDIVIDUAL FACES
# -----------------------------------------------------------------
print("\n[Section 3] Extracting individual faces...")
for i, (x, y, w, h) in enumerate(faces):
 # Extract the face region
 face_roi = image[y:y+h, x:x+w]
 
 # Display each face
 cv2.imshow(f"Face {i+1}", face_roi)
 cv2.waitKey(0)
 
 # Save each face
 cv2.imwrite(f"face_{i+1}.png", face_roi)
 print(f" Saved face {i+1} as face_{i+1}.png")
# -----------------------------------------------------------------
# SECTION 4: PARAMETER TUNING COMPARISON
# -----------------------------------------------------------------
print("\n[Section 4] Parameter tuning comparison...")
# Try different parameter combinations
param_combinations = [
 (1.05, 3, "Scale=1.05, Neighbors=3"),
 (1.1, 5, "Scale=1.1, Neighbors=5"),
 (1.2, 5, "Scale=1.2, Neighbors=5"),
 (1.1, 10, "Scale=1.1, Neighbors=10"),
]
tuning_results = []
for scale, neighbors, label in param_combinations:
 faces = detector.detect(image, scaleFactor=scale,minNeighbors=neighbors)
 
 result_img = image.copy()
 for (x, y, w, h) in faces:
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
 cv2.putText(result_img, f"{len(faces)} faces", (10, 30),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 cv2.putText(result_img, label, (10, 60),
 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
 
 tuning_results.append(result_img)
 print(f" {label}: found {len(faces)} faces")
# Create a comparison grid
display_grid = np.hstack(tuning_results)
cv2.imshow("Parameter Comparison", display_grid)
cv2.waitKey(0)
cv2.imwrite("parameter_comparison.png", display_grid)
print("\nParameter Tuning Guidelines:")
print(" - Small scaleFactor (1.05): More accurate, slower")
print(" - Large scaleFactor (1.2): Faster, may miss faces")
print(" - Small minNeighbors (3): More faces, more false positives")
print(" - Large minNeighbors (10): Fewer faces, fewer false positives")
# -----------------------------------------------------------------
# SECTION 5: DIFFERENT CASCADE TYPES
# -----------------------------------------------------------------
print("\n[Section 5] Different cascade types...")
# Try different cascade files if available
cascade_types = [
 ("Default", "haarcascade_frontalface_default.xml"),
 ("Alt", "haarcascade_frontalface_alt.xml"),
 ("Alt2", "haarcascade_frontalface_alt2.xml")
]
cascade_dir = os.path.dirname(args["face"])
cascade_results = []
for name, filename in cascade_types:
 cascade_path = os.path.join(cascade_dir, filename)
 if os.path.exists(cascade_path):
    try:
        test_detector = FaceDetector(cascade_path)
        faces = test_detector.detect(image)
 
        result_img = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0,255, 0), 2)
 
        cv2.putText(result_img, f"{name}: {len(faces)} faces", (10,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cascade_results.append(result_img)
        print(f" {name}: found {len(faces)} faces")
    except:
        print(f" {name}: could not load")
 else:
    print(f" {name}: file not found")
if cascade_results:
 display_grid = np.hstack(cascade_results[:4])
 cv2.imshow("Cascade Type Comparison", display_grid)
 cv2.waitKey(0)
# -----------------------------------------------------------------
# SECTION 6: MULTIPLE FACE DETECTION
# -----------------------------------------------------------------
print("\n[Section 6] Real-world: Multiple face detection...")
# For images with multiple faces, print each face position
if len(faces) > 1:
 print("\nFace positions in the image:")
 for i, (x, y, w, h) in enumerate(faces):
    print(f" Face {i+1}: x={x}, y={y}, width={w}, height={h}")
    print(f" Center: ({x + w//2}, {y + h//2})")
# -----------------------------------------------------------------
# SECTION 7: CREATE REFERENCE GUIDE
# -----------------------------------------------------------------
print("\n[Section 7] Creating reference guide...")
reference = np.zeros((500, 700, 3), dtype="uint8")
# Title
cv2.putText(reference, "FACE DETECTION REFERENCE GUIDE", (100, 40),
 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
# Parameters
cv2.putText(reference, "PARAMETERS:", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
0.6, (255, 255, 0), 2)
cv2.putText(reference, "scaleFactor = 1.1 (1.05-1.2)", (30, 120),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "minNeighbors = 5 (3-10)", (30, 145),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "minSize = (30, 30) (minimum face size)", (30,
170),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
# Cascade files
cv2.putText(reference, "COMMON CASCADES:", (30, 210),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.putText(reference, "haarcascade_frontalface_default.xml", (30, 240),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "haarcascade_frontalface_alt.xml", (30, 265),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "haarcascade_frontalface_alt2.xml", (30, 290),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
# Tips
cv2.putText(reference, "TIPS:", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
(255, 255, 0), 2)
cv2.putText(reference, "1. Start with default parameters", (30, 360),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "2. Adjust scaleFactor first, then minNeighbors",
(30, 385),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "3. Use grayscale images for better results", (30,
410),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(reference, "4. Smaller minSize = detect smaller faces", (30,
435),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.imshow("Face Detection Reference Guide", reference)
cv2.waitKey(0)
cv2.imwrite("face_detection_reference.png", reference)
print("\n" + "=" * 70)
print("DAY 15 COMPLETE!")
print("=" * 70)
cv2.destroyAllWindows()