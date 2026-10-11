import cv2

image = cv2.imread('image_assets/image2.jpg')

if image is None:
  print("\nError: Image not Found.")
else:
  print("\nImage Loaded Successfully!")
  cv2.imshow("Image",image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
