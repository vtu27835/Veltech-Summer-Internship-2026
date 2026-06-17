# Line 1: Import NumPy library for numerical and array operations 
import numpy as np

# Line 2: Import argparse to handle command-line arguments 
import argparse

# Line 3: Import OpenCV library for image processing tasks 
import cv2

# Line 4: Create an ArgumentParser object to handle input arguments 
ap = argparse.ArgumentParser()

# Line 5: Add a command-line argument definition for input image 
ap.add_argument(
    # Line 6: Define short (-i) and long (--image) argument names 
    "-i", "--image",
    # Line 7: Make this argument required when running the script 
    required=True,
    # Line 8: Provide help message shown in terminal when using --help 
    help="Path to the image"
# Line 9: Close the add_argument() function call 
)

# Line 10: Parse the command-line arguments into a dictionary 
args = vars(ap.parse_args())

# Line 11: Load the image from the file path provided in arguments 
image = cv2.imread(args["image"])

# Line 12: Convert the image from BGR color space to grayscale 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Line 13: Apply Gaussian blur to reduce noise and smooth the image 
image = cv2.GaussianBlur(image, (5, 5), 0)

# Line 14: Display the blurred grayscale image in a window titled "Blurred" 
cv2.imshow("Blurred", image)

# Line 15: Apply Canny edge detection with threshold values 30 and 150 
canny = cv2.Canny(image, 30, 150)

# Line 16: Display the edge-detected image in a window titled "Canny" 
cv2.imshow("Canny", canny)

# Line 17: Wait indefinitely until a key is pressed 
cv2.waitKey(0)