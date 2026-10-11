import cv2

image = cv2.imread('image_assets/image2.jpg')

if image is not None:
  print("\nImage Loaded Successfully!")
  
  cv2.imshow("Image",image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  success = cv2.imwrite("image_assets/output_image2.jpg",image)
  if success:
    print("Image Save Successfully!")
  else:
    print("Failed to Save the Image.")
    
else:
  print("\nError: Image not found.")
