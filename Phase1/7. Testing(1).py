import cv2

image = cv2.imread('image_assets/image1.jpg')

if image is not None:
  
  print("\nImage Loaded Successfully.")
  
  hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
  
  cv2.imshow('HueLightSaturation - (HLS)', hls)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  success = cv2.imwrite("image_assets/Output_Testing(1).jpg",hls)
  
  if success:
    h, w, c = hls.shape
    print(f"\nImage Details:\nHeight: {h}\nWidth: {w}\nColorChannels: {c}")
    print(f"\nImage Saved.")
  else:
    print("ERROR: Image not saved.")
  
else:
  print("ERROR: Couldn't Load the image.")
