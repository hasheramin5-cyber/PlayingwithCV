import cv2

image = cv2.imread("image_assets/image1.jpg")

if image is None:
  print("Oh no! Could not read the image.")
else:
  print("Image read successfully!")
  
  start_point = (0,145) # The exact pixel coordinate where the line starts.
  end_point = (480, 190) # The exact pixel coordinate where the line stops.
  
  '''
  
  cv2.line() are simply the (x, y) coordinates for where the line begins (pt1) and where it ends (pt2) on an image grid.
  
  • Origin (0, 0): Located at the top-left corner of the image.
  • X-axis: Starts at 0 on the left and increases as you move right.
  • Y-axis: Starts at 0 at the top and increases as you move down.
  
  
  Used Terms:
  • image: The canvas or picture you are drawing on.
  • start_point (x1, y1): The exact pixel coordinate where the line starts.
  • end_point (x2, y2): The exact pixel coordinate where the line stops.
  • color: A tuple of BGR values (Blue, Green, Red), like (255, 0, 0) for blue.
  • thickness: How wide the line is in pixels.
  
  '''
  
  color = (0, 0, 0) # Should be in BGR format
  thickness = 2 # thickness of line wanna' draw
  
  line = cv2.line(image, start_point, end_point, color, thickness)
  
  cv2.imshow("Image_Line", line)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  save = cv2.imwrite("image_assets/line_image.jpg",line)
  print("Image saved Successfully!")
