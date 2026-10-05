import cv2
import numpy as np

image = cv2.imread('image_assets/image1.jpg')

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

if image is None:
  print("Error: Image not Found.")
  
else:
  sharpen = cv2.filter2D(image, -1, sharpen_kernel)
  
  cv2.imshow("Original Image", image)
  cv2.imshow("Sharpen Image", sharpen)
  
  cv2.imwrite("image_assets/sharpening_image.jpg", sharpen)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()