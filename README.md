# ip

Program 1.	Develop a program to display grayscale image using read and write operation 
Description:
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color.
to read an image we use the function cv2.imread().
to save a image we use cv2.imwrite().
to destroy all the windows().
program:

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

output:
![image](https://user-images.githubusercontent.com/72382689/104428384-47bd0980-5539-11eb-8111-1e6a3e9a66ff.png)

******************************************************************************************************************************************************************************


Program 2:2.	Develop a program to perform linear transformation image.
The image of a linear transformation or matrix is the span of the vectors of the linear transformation. (Think of it as what vectors you can get from applying the linear transformation or multiplying the matrix by a vector.) It can be written as Im(A).
Scaling:Image scaling is the process of resizing a digital image. Scaling down an image makes it smaller while scaling up an image makes it larger. Both raster graphics and vector graphics can be scaled, but they produce different results.

Program:
import cv2 as c
img=c.imread("flower.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)
output:
![image](https://user-images.githubusercontent.com/72382689/104428763-b1d5ae80-5539-11eb-9e1e-54fc707a84b5.png)

*************************************************************************************************************************************************************************

Rotation:
This is because the rotation preserves all angles between the vectors as well as their lengths. ... Thus rotations are an example of a linear transformation by the following theorem gives the matrix of a linear transformation which rotates all vectors through an angle of θ.

import cv2 
import numpy as np 
img = cv2.imread('flower.jpg') 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('image', img)
cv2.waitKey(0) 
cv2.imshow('result',res) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


Output:
![image](https://user-images.githubusercontent.com/72382689/104430330-645a4100-553b-11eb-947e-9184490c9f07.png)
*****************************************************************************************************************************************************************************

Program 3:
Develop a program to find sum and mean of a set of images.
import cv2
import os
path = 'C:\Pictures'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

Output:
![image](https://user-images.githubusercontent.com/72382689/104431053-50630f00-553c-11eb-9574-8127cc46b16c.png)




