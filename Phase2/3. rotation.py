import cv2

image = cv2.imread("assets/image1.jpg")

if image is not None:
    # Get the dimensions of the image
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    angle = 90  # Rotate by 90 degrees
    scale = 1.0 # means zoom factor of 1 (no zoom)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    save = cv2.imwrite('assets/rotated_image.jpg', rotated_image)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    print("Error: Could not read the image.")
