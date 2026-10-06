import cv2

image = cv2.imread('image_assets/image1.jpg')
image2 = cv2.imread('image_assets/image1.jpg', cv2.IMREAD_GRAYSCALE)

if image is not None:
  canny = cv2.Canny(image, 50, 150)
  
  cv2.imshow("Original Image", image)
  cv2.imshow("GrayScale Image", image2)
  cv2.imshow("Canny Image", canny)
  
  cv2.imwrite("image_assets/canny_edge_detection.jpg", canny)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
else:
  print("Couldn't load image.")