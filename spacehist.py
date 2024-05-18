# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:35:01 2024

@author: rdx
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
os.chdir("D:")

SLIC_SPACE= 3
PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)


# Load the image in grayscale
filename= 'topanribut.png'
image = cv.imread(filename, cv.IMREAD_COLOR)
image=  cv.bitwise_not(image)

height= image.shape[0]
width= image.shape[1]
center = (width/2, height/2) 

CHANNEL= 2 # red is 'brighter as closer to background"
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, thresholded = cv.threshold(image_gray, 0, 1, cv.THRESH_OTSU) # less smear
#_, thresholded = cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)

# connected component
seed_point = (260, 25)
cv.floodFill(thresholded, None, seed_point, (255, 255, 255), loDiff=(5), upDiff=(5))
# do the superpixel here onward

def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp= cv.filter2D(img, -1, kernel)
    return(sharp)

thresholded= sharpen(thresholded)

def draw1(img): # draw the intensity
    plt.figure(dpi=300)
    plt.imshow(img)
 
def draw2(img): # draw the bitmap
    plt.figure(dpi=300)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
                   
def rottrap(img, angle):
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    hist= np.sum(dst, axis=1)  # Sum along columns to get histogram
    return (np.trapz(hist))

def rotmoment(img, angle):
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    N= cv.moments(dst)
    cx= (N['m10'] / N['m00'])
    cy= (N['m01'] / N['m00'])
    return (cx, cy)

def rotmoment2(img, angle):
    center = (img.shape[1]/2, img.shape[0]/2) 
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    N1= cv.moments(dst[:, 0:int(dst.shape[1]/2)])
    N2= cv.moments(dst[:, int(dst.shape[1]/2):dst.shape[1]])
    cy1= (N1['m01'] / N2['m00'])
    cy2= (N2['m01'] / N1['m00'])
    return (cy1, cy2)

def rottrap_img(img, angle):
    height= img.shape[0]
    width= img.shape[1]
    center = (width/2, height/2) 
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    hist= np.sum(dst, axis=1)  # Sum along columns to get histogram
    #plt.figure(dpi=300)
    #plt.imshow(dst)
    draw2(dst)
    #print(np.trapz(hist))
    return(dst)

angles= np.linspace(-6,6,200)

# check for orientation w/ moment
ax=[]
ay=[]
for i in angles:
    cx, cy= rotmoment(thresholded, i)
    ax.append(cx)
    ay.append(cy)
    
hst=[]
for i in angles:
    h= rottrap(thresholded, i)
    hst.append(h)
    
    
# find optimal page orientation, ternary search 
# page orientation usually wants to MAXIMIZE 
#  the area under the histogram curve
#left, right = -pow(PHI,3), pow(PHI,3)  # Define the range to search within
left, right = -6, 6  # Define the range to search within
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
#render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
#render= cv.cvtColor(render[:,:,CHANNEL], cv.COLOR_GRAY2RGB)
_, renderbw = cv.threshold(thresholded, 0, 240, cv.THRESH_BINARY)
render= cv.cvtColor(renderbw, cv.COLOR_GRAY2RGB)
plt.figure(dpi=300)
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
histogram_ys= gaussian_filter1d(histogram_y, pow(PHI,3))
_,valleys= find_peaks_valleys(histogram_ys)
#valleys.append(len(histogram))
valleys.insert(0,0) # append 0 at the beginning

def average_difference(lst):
    differences = [lst[i+1] - lst[i] for i in range(len(lst)-1)]
    avg_diff = sum(differences) / len(differences)
    return avg_diff

step= average_difference(valleys) / np.sqrt(PHI)

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
        # left, right = -pow(PHI,3), pow(PHI,3)  # Define the range to search within
        # epsilon = 1e-6  # Define the desired precision
        # while abs(right - left) > epsilon:
        #     mid1 = left + (right - left) / 3
        #     mid2 = right - (right - left) / 3
        #     if rottrap(linecrop, mid1) < rottrap(linecrop, mid2):
        #         right= mid2
        #     else:
        #         left= mid1
        # # found it
        # orient = (left + right) / 2
        # print(orient)
        # M = cv.getRotationMatrix2D(lcenter, orient, 1.0)
        # linecrop= cv.warpAffine(linecrop, M, (lheight,lwidth))
        
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
    elif  (histogram_x[i]<PHI*SLIC_SPACE):
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

# search the optimum orientation,
c=[]
moment_min=1e9
for i in np.linspace(-8, 8, 200):
    res= rotmoment2(linecrop,i)
    c.append(res)
    moment_i= (res[0]-res[1])
    if np.abs(moment_i) < moment_min:
        moment_min= np.abs(moment_i)
        angle_min= i
        
    
    

