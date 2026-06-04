# exercise_1.py
import cv2
image = cv2.imread("test_image.png") # Load PNG
cv2.imwrite("output.jpg", image) # Save as JPEG
cv2.imwrite("output.bmp", image) # Save as BMP
cv2.imwrite("output.tiff", image) # Save as TIFF
print("Saved in 3 formats!")