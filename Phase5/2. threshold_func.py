import cv2

image = cv2.imread('image_assets/image1.jpg')
image2 = cv2.imread('image_assets/image1.jpg', cv2.IMREAD_GRAYSCALE)

if image is not None:
  ret, thresh_img = cv2.threshold(image2, 120, 255, cv2.THRESH_BINARY)
  
  cv2.imshow("Original Image", image)
  cv2.imshow("GrayScale Image", image2)
  cv2.imshow("Threshold Image", thresh_img)
  
  cv2.imwrite("image_assets/canny_edge_detection.jpg", thresh_img)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
else:
  print("Couldn't load image.")
  
'''
90 - 0 - black
130 - 255 - white
180 - 255 - white
50 - 0 - black

image should be in GRAYSCALE - can use 'cv2.IMREAD_GRAYSCALE' for image scaling for gray, should play with it so u can know how to get the correct selected value for the threshold.

100, 120, 150
best

'''
