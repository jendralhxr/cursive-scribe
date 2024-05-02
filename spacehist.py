# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:35:01 2024

@author: rdx
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


SLIC_SPACE= 3
phi= 1.6180339887498948482 # ppl says this is a beautiful number :)


# Load the image in grayscale
filename= 'sifatline.png'
image = cv.imread(filename, cv.IMREAD_COLOR)
image=  cv.bitwise_not(image)

height= image.shape[0]
width= image.shape[1]
center = (width/2, height/2) 

CHANNEL= 2
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
#_, thresholded = cv.threshold(image_gray, 0, 1, cv.THRESH_OTSU)
_, thresholded = cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)

def rottrap(img, angle):
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    hist= np.sum(dst, axis=1)  # Sum along columns to get histogram
    return (np.trapz(hist))

def rottrap_img(img, angle):
    height= img.shape[0]
    width= img.shape[1]
    center = (width/2, height/2) 
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    hist= np.sum(dst, axis=1)  # Sum along columns to get histogram
    plt.figure(dpi=300)
    plt.imshow(dst)
    print(np.trapz(hist))
    return(dst)
 
# find optimal page orientation, ternary search 
# page orientation usually wants to MAXIMIZE 
#  the area under the histogram curve
left, right = -pow(phi,3), pow(phi,3)  # Define the range to search within
epsilon = 1e-6  # Define the desired precision
while abs(right - left) > epsilon:
    mid1 = left + (right - left) / 3
    mid2 = right - (right - left) / 3
    if rottrap(thresholded, mid1) < rottrap(thresholded, mid2):
        left = mid1
    else:
        right = mid2
# found it
orient = (left + right) / 2
print(orient)
M = cv.getRotationMatrix2D(center, orient, 1.0)
thresholded = cv.warpAffine(thresholded, M, (width,height))

# Calculate histogram
column_indices = np.arange(width)  # Generate array of column indices
row_indices= np.arange(height)
histogram_x = np.sum(thresholded, axis=0)  # Sum along columns to get histogram
histogram_y = np.sum(thresholded, axis=1)  # Sum along columns to get histogram

# line segments
plt.figure(dpi=300)
#render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
#render= cv.cvtColor(render[:,:,CHANNEL], cv.COLOR_GRAY2RGB)
_, renderbw = cv.threshold(thresholded, 0, 240, cv.THRESH_BINARY)
render= cv.cvtColor(renderbw, cv.COLOR_GRAY2RGB)
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
valleys.insert(0,0) # append 0 at the beginning

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
        linecrop= renderbw[top:bot,:]
        lwidth= linecrop.shape[0]
        lheight= linecrop.shape[1]
        lcenter= (int(lwidth/2), int(lheight/2))
        
        # correct the line orientation
        # line orientation perhaps wants to MINIMIZE
        #  the area under the histogram curve
        left, right = -pow(phi,3), pow(phi,3)  # Define the range to search within
        epsilon = 1e-6  # Define the desired precision
        while abs(right - left) > epsilon:
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            if rottrap(linecrop, mid1) < rottrap(linecrop, mid2):
                right= mid2
            else:
                left= mid1
        # found it
        orient = (left + right) / 2
        print(orient)
        M = cv.getRotationMatrix2D(lcenter, orient, 1.0)
        linecrop= cv.warpAffine(linecrop, M, (lheight,lwidth))
        
        plt.figure(dpi=300)
        plt.imshow(linecrop)
        m=m+n
        n=1 
    else:
        n= n+1


#-----------

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


