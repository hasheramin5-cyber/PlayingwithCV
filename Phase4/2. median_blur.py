import cv2

image = cv2.imread('image_assets/image1.jpg')

kernel_size = 5

if image is None:
  print("Error: Image not Found.")
  
else:
  median_blur = cv2.medianBlur(image, kernel_size)
  
  cv2.imshow("Original Image", image)
  cv2.imshow("Median Blur", median_blur)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
