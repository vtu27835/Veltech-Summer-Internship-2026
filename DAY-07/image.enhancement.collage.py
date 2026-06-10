import cv2
import numpy as np
import imutils
main_img = cv2.imread("challenge_main.png")
blend_img = cv2.imread("challenge_blend.png")

print(main_img.shape)
print(blend_img.shape)
# Create a copy for annotation
annotated = main_img.copy()

# Draw a bounding box around the blue square
cv2.rectangle(
    annotated,
    (40, 40),
    (160, 160),
    (0, 255, 255),
    3
)

# Add a text label
cv2.putText(
    annotated,
    "Blue Square",
    (40, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

# Highlight the green circle
cv2.circle(
    annotated,
    (300, 120),
    70,
    (255, 255, 255),
    3
)

# Display the result
cv2.imshow("Annotated Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Create a rotated version
transformed = imutils.rotate(main_img, 30)

# Display it
cv2.imshow("Transformed Image", transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Blend the two images
blended = cv2.addWeighted(
    main_img,
    0.7,
    blend_img,
    0.3,
    0
)

# Display blended image
cv2.imshow("Blended Image", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Create a black mask
mask = np.zeros(main_img.shape[:2], dtype="uint8")

# Draw a white filled circle in the center
cv2.circle(
    mask,
    (200, 200),
    120,
    255,
    -1
)

# Apply the mask
masked = cv2.bitwise_and(
    main_img,
    main_img,
    mask=mask
)

# Show mask and result
cv2.imshow("Mask", mask)
cv2.imshow("Masked Result", masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Resize images for collage
annotated_small = cv2.resize(annotated, (300, 300))
transformed_small = cv2.resize(transformed, (300, 300))
blended_small = cv2.resize(blended, (300, 300))
masked_small = cv2.resize(masked, (300, 300))
top_row = np.hstack([
    annotated_small,
    transformed_small
])
bottom_row = np.hstack([
    blended_small,
    masked_small
])
collage = np.vstack([
    top_row,
    bottom_row
])
cv2.imwrite("annotated.png", annotated)
cv2.imwrite("transformed.png", transformed)
cv2.imwrite("blended.png", blended)
cv2.imwrite("masked.png", masked)
cv2.imwrite("final_collage.png", collage)
cv2.imshow("Final Collage", collage)
cv2.waitKey(0)
cv2.destroyAllWindows()