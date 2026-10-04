import cv2

image = cv2.imread("assets/image1.jpg")

if image is not None:

    flip_code = 0  # Flip vertically (0), horizontally (1), or both (-1)
    
    flipped_image = cv2.flip(image, flip_code)
    
    save = cv2.imwrite('assets/flipped_image.jpg', flipped_image)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Flipped Image', flipped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    print("Error: Could not read the image.")
    
