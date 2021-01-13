# ip

Program 1.	Develop a program to display grayscale image using read and write operation 
Description:
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color.
to read an image we use the function cv2.imread().
to save a image we use cv2.imwrite().
to destroy all the windows().
import cv2
img = cv2.imread('flower.jpg')
cv2.imshow('Input',img)
cv2.waitKey(0)
grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscaleimage',grayimg)
cv2.waitKey(0)
ret, bw_img = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.imwrite("gray.jpg",grayimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

