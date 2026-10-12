import cv2

image = cv2.imread('image_assets/image1.jpg')
# width = 800 and then height = 600

if image is None:
  print("Error: Could not read the image.")
else:
  print("Image read successfully.")
  
  resized_image = cv2.resize(image, (800, 600))
  
  save = cv2.imwrite('image_assets/resized_image.jpg', resized_image)
  
  cv2.imshow('Original Image', image)
  cv2.imshow('Resized Image', resized_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
    