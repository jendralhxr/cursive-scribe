# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:35:01 2024

@author: rdx
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('sbk.png', cv2.IMREAD_COLOR)
image=  cv2.bitwise_not(image)

height= image.shape[0]
width= image.shape[1]

image_gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, thresholded = cv2.threshold(image_gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Calculate histogram
num_cols = thresholded.shape[1]
column_indices = np.arange(num_cols)  # Generate array of column indices
histogram = np.sum(thresholded, axis=0)  # Sum along columns to get histogram

SLIC_SPACE= 3

for i in range(width):
    if (histogram[i]<SLIC_SPACE):
        for j in range(height):
            image.itemset((j,i,2), 255)

image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) 



# Plot the histogram
plt.plot(column_indices, histogram)
plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
plt.xlabel('Column Index')
plt.ylabel('Number of Pixels')
plt.show()