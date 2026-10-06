'''
1- cv2.bitwise_and(img1, img2)
2- cv2.bitwise_or(img1, img2)
3- cv2.bitwise_not(img1)

* img1 ,img2 height and width same
** use only black and white color
*** will also use NumPy - (import numpy as np)
**** 

'''
import cv2
import numpy as np

img1 = np.zeros((300, 300), dtype="uint8")
img2 = np.zeros((300, 300), dtype="uint8")

cv2.circle(img1, (150, 150), 100, 255, -1)
cv2.rectangle(img2, (100, 100), (250, 250), 255, -1)

bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_not_circle = cv2.bitwise_not(img1)
bitwise_not_rectangle = cv2.bitwise_not(img2)

cv2.imshow("Circle", img1)
cv2.imshow("Rectangle", img2)

cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("NOT CIRCLE", bitwise_not_circle)
cv2.imshow("NOT RECTANGLE", bitwise_not_rectangle)

cv2.waitKey(0)
cv2.destroyAllWindows()