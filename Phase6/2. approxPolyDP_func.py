import cv2
import numpy as np


image = cv2.imread(r'image_assets\shape.jpg')

if image is None:
    raise FileNotFoundError(
        "Image not found. Check that image_assets\\shape.jpg exists."
    )

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Use inverse threshold if shapes are dark and the background is white
_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)


# Find contours
contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


for contour in contours:

    # Ignore very small noise
    if cv2.contourArea(contour) < 100:
        continue

    epsilon = 0.01 * cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    corners = len(approx)

    if corners == 3:
        shape_name = "Triangle"
    elif corners == 4:
        shape_name = "Rectangle"
    elif corners == 5:
        shape_name = "Pentagon"
    elif corners > 5:
        shape_name = "Circle"
    else:
        shape_name = "Unknown"

    cv2.drawContours(
        image,
        [approx],
        0,
        (0, 255, 0),
        2
    )

    x = approx.ravel()[0]
    y = approx.ravel()[1] - 10

    cv2.putText(
        image,
        shape_name,
        (x, y),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


cv2.imshow("Threshold Image", thresh)
cv2.imshow("Detected Shapes", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
