import cv2

image = cv2.imread('assets/image1.jpg')

if image is not None:
  hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
  cv2.imshow('HueLightSaturation - (HLS)', hls)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
else:
  print("Couldn't Load the image.")
  