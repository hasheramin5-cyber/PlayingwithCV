
import cv2
import numpy as np

image = cv2.imread("RGB_Colors/input.jpg")

if image is None:
    raise FileNotFoundError("Image not found!")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hsv[:, :, 0] = 60
hsv[:, :, 1] = 255

green_filter = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
result = cv2.addWeighted(image, 0.0, green_filter, 1.0, 0)

cv2.imwrite("RGB_Colors/input_green_filter.jpg", result)
print("Green colorize filter applied!")
