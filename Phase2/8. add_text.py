import cv2

image = cv2.imread('image_assets/image4.jpg')

if image is None:
  print("Oh no! Could not read the image.")
else:
  print("Image read successfully!")
  
  text = 'Hello Clove!'
  org = (20, 190)
  font = cv2.FONT_HERSHEY_COMPLEX_SMALL
  fontScale = 1
  color = (85, 97, 135)
  thickness = 2
  
  text = cv2.putText(image, text, org, font, fontScale, color, thickness)
  
  cv2.imshow("Text Image", text)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  save = cv2.imwrite("image_assets/text_image.jpg",text)
  print("Image saved Successfully!") 