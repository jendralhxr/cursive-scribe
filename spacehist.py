# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:35:01 2024

@author: rdx
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
filename= 'sifatline.png'
image = cv2.imread(filename, cv2.IMREAD_COLOR)
image=  cv2.bitwise_not(image)

height= image.shape[0]
width= image.shape[1]
center=(width/2, height/2)

CHANNEL= 2
image_gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, thresholded = cv2.threshold(image_gray, 0, 1, cv2.THRESH_OTSU)

# Calculate histogram
column_indices = np.arange(width)  # Generate array of column indices
row_indices= np.arange(height)
histogram_x = np.sum(thresholded, axis=0)  # Sum along columns to get histogram
histogram_y = np.sum(thresholded, axis=1)  # Sum along columns to get histogram

SLIC_SPACE= 3
phi= 1.6180339887498948482 # ppl says this is a beautiful number :)

# line segments
plt.figure(dpi=300)
render= thresholded.copy()
#render= cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
render= cv2.cvtColor(render[:,:,CHANNEL], cv2.COLOR_GRAY2RGB)
plt.imshow(render) 
plt.plot(histogram_y, row_indices)
#plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
plt.xlabel('Row Index')
plt.ylabel('count')
plt.show()

# rotate
DEGREE= 1
M = cv2.getRotationMatrix2D(center, DEGREE, 1.0)
dst = cv2.warpAffine(thresholded, M, (width,height))
integral = np.trapz(histogram_x)


# spaces, horizontal
render= image.copy()
for i in range(width-1, -1, -1):
    if (histogram_x[i]<=SLIC_SPACE):
        for j in range(height):
            render.itemset((j,i,2), 240)
    elif  (histogram_x[i]<phi*SLIC_SPACE):
        for j in range(height):
            render.itemset((j,i,2), 80)
    
render= cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
plt.figure(dpi=300)
plt.imshow(render) 
# Plot the histogram
plt.plot(column_indices, histogram_x)
#plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
plt.xlabel('Column Index')
plt.ylabel('Number of Pixels')
plt.show()


