import cv2

image = cv2.imread('image_assets/image1.jpg')
if image is not None:
  
  cropped = image[200:300 , 130:170]
  
  save = cv2.imwrite('image_assets/cropped_image.jpg', cropped)
  
  cv2.imshow('Cropped Image', cropped)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
else:
  print("Error: Could not read the image.")
