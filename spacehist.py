# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:35:01 2024

@author: rdx
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
filename= 'sifatline.png'
image = cv.imread(filename, cv.IMREAD_COLOR)
image=  cv.bitwise_not(image)

height= image.shape[0]
width= image.shape[1]
center=(width/2, height/2)

CHANNEL= 2
image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
#_, thresholded = cv.threshold(image_gray, 0, 1, cv.THRESH_OTSU)
_, thresholded = cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)

def rottrap(angle):
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(thresholded, M, (width,height))
    hist= np.sum(dst, axis=1)  # Sum along columns to get histogram
    return np.trapz(hist)

# find optimal orientation
left, right = -1, 1  # Define the range to search within
epsilon = 1e-6  # Define the desired precision
while abs(right - left) > epsilon:
    mid1 = left + (right - left) / 3
    mid2 = right - (right - left) / 3
    if rottrap(mid1) < rottrap(mid2):
        right = mid2
    else:
        left = mid1
# found it
min_x = (left + right) / 2
#print("Minimum value:", min_value)
#print("Corresponding x:", min_x)
#min_value = rottrap(min_x)
M = cv.getRotationMatrix2D(center, min_x, 1.0)
thresholded = cv.warpAffine(thresholded, M, (width,height))

# Calculate histogram
column_indices = np.arange(width)  # Generate array of column indices
row_indices= np.arange(height)
histogram_x = np.sum(thresholded, axis=0)  # Sum along columns to get histogram
histogram_y = np.sum(thresholded, axis=1)  # Sum along columns to get histogram

SLIC_SPACE= 3
phi= 1.6180339887498948482 # ppl says this is a beautiful number :)

# line segments
plt.figure(dpi=300)
#render= thresholded.copy()
#render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
#render= cv.cvtColor(render[:,:,CHANNEL], cv.COLOR_GRAY2RGB)
_, render = cv.threshold(thresholded, 0, 240, cv.THRESH_BINARY)
render= cv.cvtColor(render, cv.COLOR_GRAY2RGB)
plt.imshow(render) 
plt.plot(histogram_y, row_indices)
#plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
plt.xlabel('Row Index')
plt.ylabel('count')
plt.show()


def find_peaks_valleys(hst):
    peaks = []
    valleys = []
    for i in range(1, len(hst) - 1):
        if hst[i] > hst[i - 1] and hst[i] >= hst[i + 1]:
            peaks.append(i)
        elif hst[i] < hst[i - 1] and hst[i] <= hst[i + 1]:
            valleys.append(i)
        elif hst[i] < hst[i - 1] and i==len(hst)-2:
            valleys.append(i)    
    return peaks, valleys

from scipy.ndimage import gaussian_filter1d
histogram_ys= gaussian_filter1d(histogram_y, pow(phi,3))
_,valleys= find_peaks_valleys(histogram_ys)
#valleys.append(len(histogram))

def average_difference(lst):
    differences = [lst[i+1] - lst[i] for i in range(len(lst)-1)]
    avg_diff = sum(differences) / len(differences)
    return avg_diff

step= average_difference(valleys) / np.sqrt(phi)

# gonna crop-select each lines here
m=0
n=1
while (valleys[m]<=max(valleys) and (m+n)<len(valleys)):
    top=valleys[m]
    bot=valleys[m+n]
    if (bot-top)>step:
        linecrop= thresholded[top:bot,:]
        plt.figure(dpi=300)
        plt.imshow(linecrop)
        m=m+n
        n=1 
    else:
        n= n+1
    
# spaces, horizontal
render= image.copy()
for i in range(width-1, -1, -1):
    if (histogram_x[i]<=SLIC_SPACE):
        for j in range(height):
            render.itemset((j,i,2), 240)
    elif  (histogram_x[i]<phi*SLIC_SPACE):
        for j in range(height):
            render.itemset((j,i,2), 80)
    
render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
plt.figure(dpi=300)
plt.imshow(render) 
# Plot the histogram
plt.plot(column_indices, histogram_x)
#plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
plt.xlabel('Column Index')
plt.ylabel('Number of Pixels')
plt.show()


