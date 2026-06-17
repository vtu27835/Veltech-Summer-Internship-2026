# Line 1: Import NumPy for numerical operations and array handling
import numpy as np
# Line 2: Import argparse to read command-line arguments
import argparse
# Line 3: Import OpenCV for image processing
import cv2
# Line 5: Create an argument parser object
ap = argparse.ArgumentParser()
# Lines 6-7: Define a required command-line argument (-i or --image)
# that specifies the path to the input image
ap.add_argument(
 "-i",
 "--image",
 required=True,
 help="Path to the image"
)
# Line 8: Parse the command-line arguments and convert them into a dictionary
args = vars(ap.parse_args())
# Line 10: Load the image from the specified file path
image = cv2.imread(args["image"])
# Line 11: Convert the loaded color image (BGR) to grayscale
# because edge detection works more effectively on single-channel images
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Line 12: Display the grayscale image in a window titled "Original"
cv2.imshow("Original", image)
# Line 14: Apply the Laplacian operator to detect edges
# CV_64F is used to store both positive and negative gradient values
lap = cv2.Laplacian(image, cv2.CV_64F)
# Line 15: Take the absolute value of the Laplacian result
# and convert it to an unsigned 8-bit image for display
lap = np.uint8(np.absolute(lap))
# Line 16: Display the edge-detected image in a window titled "Laplacian"
cv2.imshow("Laplacian", lap)
# Line 17: Wait indefinitely until a key is pressed
# This prevents the windows from closing immediately
cv2.waitKey(0)
# Line 18: Apply the Sobel operator to detect edges in the x-direction (horizontal edges)
# The parameters (1, 0) indicate differentiation along the x￾axis only
# CV_64F allows capturing negative gradient values.
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
# Line 19: Apply the Sobel operator to detect edges in the y-direction (vertical edges)
# The parameters (0, 1) indicate differentiation along the yaxis only
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
# Line 20: Blank line for readability
# Line 21: Convert Sobel X results to absolute values and then to 8-bit unsigned integers
# This makes the image suitable for display
sobelX = np.uint8(np.absolute(sobelX))
# Line 22: Convert Sobel Y results to absolute values and then to 8-bit unsigned integers
sobelY = np.uint8(np.absolute(sobelY))
# Line 23: Blank line for readability
# Line 24: Combine the horizontal and vertical Sobel edge images using a bitwise OR operation
# This produces a single image containing all detected edges
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
# Line 25: Blank line for readability
# Line 26: Display the horizontal edge image in a window titled "Sobel X"
cv2.imshow("Sobel X", sobelX)
# Line 27: Display the vertical edge image in a window titled "Sobel Y"
cv2.imshow("Sobel Y", sobelY)
# Line 28: Display the combined edge image in a window titled "Sobel Combined"
cv2.imshow("Sobel Combined", sobelCombined)
# Line 29: Wait indefinitely until a key is pressed before closing the image windows
cv2.waitKey(0)