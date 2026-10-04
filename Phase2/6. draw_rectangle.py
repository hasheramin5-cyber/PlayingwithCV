import cv2

image = cv2.imread('assets/image1.jpg')

if image is None:
  print("Oh no! Could not read the image.")
else:
  print("Image read successfully!")

  start_point = (175, 300)
  end_point = (125, 170)
  color = (0, 0, 0)
  thickness = 2
  
  rectangle = cv2.rectangle(image, start_point, end_point, color, thickness)
  
  cv2.imshow("Draw Rectangle", rectangle)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  save = cv2.imwrite("assets/rectangle_image.jpg",rectangle)
  print("Image saved Successfully!")