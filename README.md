# Image Processing

**Program 1.	Develop a program to display grayscale image using read and write operation**

**Description:**

Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color.
to read an image we use the function cv2.imread().
to save a image we use cv2.imwrite().
to destroy all the windows().
program:

```python
import numpy as np
import cv2
image=cv2.imread('flower.jpg',0)
cv2.imshow('Original',image)
image=cv2.imread('flower.jpg')
cv2.imshow('Gray',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("grayscale.png",image)
```

output:
![image](https://user-images.githubusercontent.com/72382689/105327393-b296c200-5b83-11eb-901f-ce0feeb40c42.png)

**------------------------------------------------------------------------------------------------------------------------------------------------------------**

**Program 2:	Develop a program to perform linear transformation image.**

**Description:**

The image of a linear transformation or matrix is the span of the vectors of the linear transformation. (Think of it as what vectors you can get from applying the linear transformation or multiplying the matrix by a vector.) It can be written as Im(A).
Scaling:Image scaling is the process of resizing a digital image. Scaling down an image makes it smaller while scaling up an image makes it larger. Both raster graphics and vector graphics can be scaled, but they produce different results.

```python
Program:
import cv2 as c
img=c.imread("flower.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)
```


**output:**
![image](https://user-images.githubusercontent.com/72382689/105327864-48cae800-5b84-11eb-9e90-f3b1a8683de8.png)


**------------------------------------------------------------------------------------------------------------------------------------------------------------**

**Rotation:**


This is because the rotation preserves all angles between the vectors as well as their lengths.
Thus rotations are an example of a linear transformation by the following theorem gives the matrix of a linear transformation which rotates all vectors through an angle of θ.

```python
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
```

**Output:**
![image](https://user-images.githubusercontent.com/72382689/105328568-05bd4480-5b85-11eb-9707-ad4a503d862e.png)


**-----------------------------------------------------------------------------------------------------------------------------------------------------------**


**Program 3:Develop a program to find sum and mean of a set of images.**

**Description:**

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

**-----------------------------------------------------------------------------------------------------------------------------------------------------------**

**Program 4: Write a program to convert color image into gray scale and binary image**

**Description:**

Color image to Grayscale image
The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. ... In addition, grayscale images are entirely sufficient for many tasks and so there is no need to use more complicated and harder-to-process color images.
Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color.

```python
import cv2
image=cv2.imread("butterfly.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(tresh,blackAndWhiteImage)=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("gray",gray)
cv2.imshow("BINARY",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Output:**

![image](https://user-images.githubusercontent.com/72382689/105333134-3a7fca80-5b8a-11eb-8d1d-ba48703bbe36.png)



**-----------------------------------------------------------------------------------------------------------------------------------------------------------**


**Program 5: Write a program to convert color image into different color space.**

**Description:**

A color space is a specific organization of colors. In combination with color profiling supported by various physical devices, and supports reproducible representations of color -- whether such representation entails an analog or a digital representation.
COLOR_BGR2HSV which is used to change BGR image to HSV image.
COLOR_BGR2LAB which is used to change BGR image to LAB image.
COLOR_BGR2HLS which is used to change BGR image to HLS image.

```python
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
```

**Output:**

![image](https://user-images.githubusercontent.com/72382689/105333460-9a767100-5b8a-11eb-8af0-17f72a28d4b9.png)


**-----------------------------------------------------------------------------------------------------------------------------------------------------------**


**Program 6: Develop a program to create an image from 2D array.**

**Description:**

An image is a visual representation of something. ... 1) An image is a picture that has been created or copied and stored in electronic form. An image can be described in terms of vector graphics or raster graphics. An image stored in raster form is sometimes called a bitmap.
An image is an array, or a matrix, of square pixels (picture elements) arranged in columns and rows. An image — an array or a matrix of pixels arranged in columns and rows. In a (8-bit) greyscale image each picture element has an assigned intensity that ranges from 0 to 255.
A 2D array has a type such as int[][] or String[][], with two pairs of square brackets. ... The elements of a 2D array are arranged in rows and columns, and the new operator for 2D arrays specifies both the number of rows and the number of columns.

```python
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
```

**Output:**
![image](https://user-images.githubusercontent.com/72382689/104434760-768aae00-5540-11eb-9fb3-13910318a046.png)

**-----------------------------------------------------------------------------------------------------------------------------------------------------------**



**Program 7: Find the sum of the neighborhood values of the matrix.**


**Description:**

```python
import numpy as np

M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 

M = np.asarray(M)
N = np.zeros(M.shape)

def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)

print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)
```

**Output:**


Original matrix:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
Summed neighbors matrix:
 [[11. 19. 13.]
 [23. 40. 27.]
 [17. 31. 19.]]
 
 
 **-----------------------------------------------------------------------------------------------------------------------------------------------------------**

**Program 8: Find the nieghborhood values of the matrix**


**Description**


```python
import numpy as np
ini_array = np.array([[1, 2,5, 3], [4,5, 4, 7], [9, 6, 1,0]])
print("initial_array : ", str(ini_array));
def neighbors(radius, rowNumber, columnNumber):
    return[[ini_array[i][j]if i >= 0 and i < len(ini_array) and j >= 0 and j < len(ini_array[0]) else 0
            for j in range(columnNumber-1-radius, columnNumber+radius)]
           for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(2, 2, 2)
```
**Output:**

initial_array :  [[1 2 5 3]
 [4 5 4 7]
 [9 6 1 0]]
[[0, 0, 0, 0, 0],
 [0, 1, 2, 5, 3],
 [0, 4, 5, 4, 7],
 [0, 9, 6, 1, 0],
 [0, 0, 0, 0, 0]]
 
**-----------------------------------------------------------------------------------------------------------------------------------------------------------**
**Program 9: Operator Overloading**

**Description**

```cpp
#include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];

   }
  }
 
 
 };
 void operator+(matrix a1)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
   
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }

 };

  void operator-(matrix a2)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }
   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };

 void operator*(matrix a3)
 {
  int c[i][j];

  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
 };

};

int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}
```

**Output**

Enter the row and column size for the  matrix                                                                                   
2                                                                                                                               
2                                                                                                                               
Enter the elements of the matrix                                                                                                
6                                                                                                                               
7                                                                                                                               
8                                                                                                                               
9                                                                                                                               
Enter the row and column size for the  matrix                                                                                   
2                                                                                                                               
2                                                                                                                               
Enter the elements of the matrix                                                                                                
1                                                                                                                               
2                                                                                                                               
3                                                                                                                               
4                                                                                                                               
addition is                                                                                                                     
 7      9                                                                                                                       
 11     13                                                                                                                      
subtraction is                                                                                                                  
 5      5                                                                                                                       
 5      5                                                                                                                       
multiplication is                                                                                                               
 27     40                                                                                                                      
 35     52       

 **-----------------------------------------------------------------------------------------------------------------------------------------------------------**
 
 **Program 10: Develop a program to implement negative transformation**
 
 
 **Description**
 
 
 ```python
 import cv2
import numpy as np
img=cv2.imread("flower.jpg")
cv2.imshow("original",img)
cv2.waitKey(0)
img_neg=255-img
cv2.imshow('negative',img_neg)
cv2.waitKey(0)
```


**Output:**

![image](https://user-images.githubusercontent.com/72382689/105163180-718da780-5ac8-11eb-826e-8dca91e5bc18.png)

**-----------------------------------------------------------------------------------------------------------------------------------------------------------**


**Program 11: Develop a program to implement Contrast transformation**


**Description:**


```python
from PIL import Image, ImageEnhance
im = Image.open(r"flower.jpg")
im.show()
im3 = ImageEnhance.Color(im)
im3.enhance(4.3).show()
```

**Output:**
![image](https://user-images.githubusercontent.com/72382689/105328239-b119c980-5b84-11eb-9965-a30b822105f8.png)

**-----------------------------------------------------------------------------------------------------------------------------------------------------------**


**Program 12: Develop a program to implement Threshold transformation**


**Description:**

```python
# Python programe to illustrate 
# simple thresholding type on an image 
	
# organizing imports 
import cv2 
import numpy as np 

# path to input image is specified and 
# image is loaded with imread command 
image1 = cv2.imread('flower.jpg') 

# cv2.cvtColor is applied over the 
# image input with applied parameters 
# to convert the image in grayscale 
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 

# applying different thresholding 
# techniques on the input image 
# all pixels value above 120 will 
# be set to 255 
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC) 
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO) 
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV) 

# the window showing output images 
# with the corresponding thresholding 
# techniques applied to the input images 
cv2.imshow('Binary Threshold', thresh1) 
cv2.imshow('Binary Threshold Inverted', thresh2) 
cv2.imshow('Truncated Threshold', thresh3) 
cv2.imshow('Set to 0', thresh4) 
cv2.imshow('Set to 0 Inverted', thresh5) 
	
# De-allocate any associated memory usage 
if cv2.waitKey(0) & 0xff == 27: 
	cv2.destroyAllWindows() 
```

**Output:**
![image](https://user-images.githubusercontent.com/72382689/105164150-a2baa780-5ac9-11eb-8985-164d5ae190e3.png)

**-----------------------------------------------------------------------------------------------------------------------------------------------------------**

**Program 13: Develop a program to implement a powerlow transformation.**


**Description:**

```python
import cv2
import numpy as np
img = cv2.imread('flower.jpg')
cv2.imshow("original",img)
for gamma in [0.1, 0.5, 1.2, 2.2]:
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype='uint8')
    cv2.imshow('gamma_transformed'+str(gamma)+'.jpg', gamma_corrected)
cv2.waitKey(0)
```

**Output:**

![image](https://user-images.githubusercontent.com/72382689/105204613-612e0900-5b6a-11eb-8512-2a3d11c90503.png)
**-----------------------------------------------------------------------------------------------------------------------------------------------------------**

**Program 14: Program to enhance image using image arithmatic and logic operations**


**Description:**


```python
# Python programe to illustrate 
# arithmetic operation of 
# addition of two images 
	
# organizing imports 
import cv2 
import numpy as np 
	
# path to input images are specified and 
# images are loaded with imread command 
image1 = cv2.imread('input1.jpg') 
image2 = cv2.imread('input2.jpg') 
cv2.imshow("Original1",image1)
cv2.imshow("Original2",image2)
# cv2.addWeighted is applied over the 
# image inputs with applied parameters 
weightedSum = cv2.add(image1,image2) 

# the window showing output image 
# with the weighted sum 
cv2.imshow('Weighted Image', weightedSum) 
sub = cv2.subtract(image1, image2)
cv2.imshow('Subtracted Image', sub) 
# De-allocate any associated memory usage 
cv2.waitKey(0)
cv2.destroyAll()
```

**Output:**
![image](https://user-images.githubusercontent.com/72382689/107620830-56532980-6c7b-11eb-892f-ab4da8045b3d.png)




