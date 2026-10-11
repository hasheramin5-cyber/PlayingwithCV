import cv2

image = cv2.imread('image_assets/image1.jpg')

if image is not None:
  h, w, c = image.shape
  print(f"Height: {h}\nWidth: {w}\nColorChannel: {c}")
else:
  print("Couldn't load the image.")
