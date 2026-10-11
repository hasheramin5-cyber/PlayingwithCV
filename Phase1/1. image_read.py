import cv2

image = cv2.imread('image_assets/image1.jpg')

if image is None:
  print("\nError: Image not Found.")
else:
  print("\nImage Loaded Successfully!")
