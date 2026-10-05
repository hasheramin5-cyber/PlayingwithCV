import cv2

kernel_size_X = 21
kernel_size_Y = 21 # Super Duper - Blur
sigma = 0

image = cv2.imread('image_assets/image5.jpg')
print(f"\nOriginal (Image Details): {image.shape}")

effect = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
print(f"Effect (Image Details): {effect.shape}")

Gaussian_Blur = cv2.GaussianBlur(effect, (kernel_size_X, kernel_size_Y), sigma)
print(f"Effect + Blur (Image Details): {Gaussian_Blur.shape}")

cv2.imshow("Original Image", image)
cv2.imshow("Effect Image", effect)
cv2.imshow("Effect + Blur Image",Gaussian_Blur)

cv2.waitKey(0)
cv2.destroyAllWindows()