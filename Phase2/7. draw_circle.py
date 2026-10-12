import cv2

image = cv2.imread('image_assets/image1.jpg')

if image is None:
  print("Oh no! Could not read the image.")
else:
  print("Image read successfully!")
  
  center = (150, 245)
  radius = 40
  color = (255, 255, 0)
  thickness = 2
  
  circle = cv2.circle(image, center, radius, color, thickness)
  
  cv2.imshow("Draw Circle", circle)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  save = cv2.imwrite("image_assets/circle_image.jpg",circle)
  print("Image saved Successfully!")

