# Image Processing

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


Program 2:	Develop a program to perform linear transformation image.

Description:

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
This is because the rotation preserves all angles between the vectors as well as their lengths.
Thus rotations are an example of a linear transformation by the following theorem gives the matrix of a linear transformation which rotates all vectors through an angle of θ.

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

Program 3:Develop a program to find sum and mean of a set of images.

Description:

Given a List. The task is to find the sum and average of the list. Average of the list is defined as the sum of the elements divided by the number of the elements.
Mean—often simply called the "average"—is a term used in statistics and data analysis. In addition, it's not unusual to hear the words "mean" or "average" used with the terms "mode," "median," and "range," which are other methods of calculating the patterns and common values in data sets.
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

*************************************************************************************************************************************************************************
Program 4: Write a program to convert color image into gray scale and binary image

Description:

Color image to Grayscale image
The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. ... In addition, grayscale images are entirely sufficient for many tasks and so there is no need to use more complicated and harder-to-process color images.
Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color.

Program:
import cv2
image=cv2.imread("butterfly.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(tresh,blackAndWhiteImage)=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("gray",gray)
cv2.imshow("BINARY",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


Output:

![image](https://user-images.githubusercontent.com/72382689/104433027-74275480-553e-11eb-8fa2-66ff5737f5e2.png)



*********************************************************************************************************************************************************************

Program 5:
Write a program to convert color image into different color space.

Description:A color space is a specific organization of colors. In combination with color profiling supported by various physical devices, and supports reproducible representations of color -- whether such representation entails an analog or a digital representation.
COLOR_BGR2HSV which is used to change BGR image to HSV image.
COLOR_BGR2LAB which is used to change BGR image to LAB image.
COLOR_BGR2HLS which is used to change BGR image to HLS image.

import cv2
image=cv2.imread("butterfly.jpg")
cv2.imshow("old",image)
cv2.waitKey()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",hsv)
cv2.waitKey(0)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB",lab)
cv2.waitKey(0)
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS",hls)
cv2.waitKey(0)
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imshow("YUV",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()


Output:

![image](https://user-images.githubusercontent.com/72382689/104434029-9bcaec80-553f-11eb-8095-a1de24232496.png)


*********************************************************************************************************************************

Program 6:
Develop a program to create an image from 2D array.

Description:

An image is a visual representation of something. ... 1) An image is a picture that has been created or copied and stored in electronic form. An image can be described in terms of vector graphics or raster graphics. An image stored in raster form is sometimes called a bitmap.
An image is an array, or a matrix, of square pixels (picture elements) arranged in columns and rows. An image — an array or a matrix of pixels arranged in columns and rows. In a (8-bit) greyscale image each picture element has an assigned intensity that ranges from 0 to 255.
A 2D array has a type such as int[][] or String[][], with two pairs of square brackets. ... The elements of a 2D array are arranged in rows and columns, and the new operator for 2D arrays specifies both the number of rows and the number of columns.


import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [150, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255]   #Blue right side
img = Image.fromarray(array)
img.save('flower.jpg')
img.show()
c.waitKey(0)


Output:
![image](https://user-images.githubusercontent.com/72382689/104434760-768aae00-5540-11eb-9fb3-13910318a046.png)

*******************z




