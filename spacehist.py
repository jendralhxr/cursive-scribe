# USAGE: python -u spacehist.py <image-of-page>

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

SLIC_SPACE= 3
PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

def draw1(img): # draw the intensity
    plt.figure(dpi=300)
    plt.imshow(img)
 
def draw2(img): # draw the bitmap
    plt.figure(dpi=300)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
                   
def rotmoment2(img, angle):
    center = (img.shape[1]/2, img.shape[0]/2) 
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    N1= cv.moments(dst[:, 0:int(dst.shape[1]/2)])
    N2= cv.moments(dst[:, int(dst.shape[1]/2):dst.shape[1]])
    cy1= (N1['m01'] / N2['m00'])
    cy2= (N2['m01'] / N1['m00'])
    return (cy1, cy2)

def rotimg(img, angle):
    height= img.shape[0]
    width= img.shape[1]
    center = (width/2, height/2) 
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    dst = cv.warpAffine(img, M, (width,height))
    draw2(dst)
    return(dst)

from scipy.ndimage import gaussian_filter1d

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


rect_out=[]
rect_fit=[]

filename= sys.argv[1]
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename, cv.IMREAD_COLOR)
image=  cv.bitwise_not(image)

height= image.shape[0]
width= image.shape[1]
center = (width/2, height/2) 

CHANNEL= 2 # red is 'brighter as closer to background"
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray= image[:,:,CHANNEL]
_, thresholded = cv.threshold(gray, 0, 1, cv.THRESH_OTSU) # less smear
_, gray = cv.threshold(gray, 0, 80, cv.THRESH_OTSU) # less smear
#_, gray = cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)

# # 'outer' bounding box
# x, y, w, h = cv.boundingRect(gray)
# r1= cv.boundingRect(gray)
# rect_out.append(r1)
# cv.rectangle(image, (x+1, y+1), (x+w-2, y+h-2), (0, 255, 0), 2)  # Draw a green rectangle with thickness 2
# # 'inner' bounding box
# contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# all_points = np.vstack(contours)
# hull = cv.convexHull(all_points)
# #cv.drawContours(image, [hull], 0, (0, 0, 255), 2)
# r2= cv.minAreaRect(all_points)
# box = cv.boxPoints(r2)
# box = np.int0(box)
# cv.drawContours(image,[box],0,(0,0,255),2)
# rect_fit.append(r2)
# draw2(image)

# # search the optimum orientation,
# c=[]
# moment_min=1e9
# angles= np.linspace(-8,8,200)
# for i in angles:
#     res= rotmoment2(gray,i)
#     c.append(res)
#     moment_i= (res[0]-res[1])
#     if np.abs(moment_i) < moment_min:
#         moment_min= np.abs(moment_i)
#         angle_min= i

# calculate histogram
column_indices = np.arange(width)  # Generate array of column indices
row_indices= np.arange(height)
histogram_x = np.sum(thresholded, axis=0)  # Sum along columns to get histogram
histogram_y = np.sum(thresholded, axis=1)  # Sum along columns to get histogram

#render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
#render= cv.cvtColor(render[:,:,CHANNEL], cv.COLOR_GRAY2RGB)
_, renderbw = cv.threshold(thresholded, 0, 240, cv.THRESH_BINARY)
render= cv.cvtColor(renderbw, cv.COLOR_GRAY2RGB)
plt.figure(dpi=300)
plt.imshow(render) 
plt.plot(histogram_y, row_indices)
plt.xlabel('row index')
plt.ylabel('bin')
plt.savefig(imagename+'-hist.png')
#plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
#plt.show()

# smooth the histogram
histogram_ys= gaussian_filter1d(histogram_y, pow(PHI,3))

# find the positions of the line segments
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
        linecrop= gray[top:bot,:]
        
        # correct the line orientation
        c=[]
        moment_min=1e9
        angles= np.linspace(-6,6,200)
        for i in angles:
            res= rotmoment2(linecrop,i)
            c.append(res)
            moment_i= (res[0]-res[1])
            if np.abs(moment_i) < moment_min:
                moment_min= np.abs(moment_i)
                angle_min= i     
                
        cv.imwrite(imagename+'-line'+str(m)+'.png', linecrop)
        linecrop= rotimg(linecrop, angle_min)
        #cv.imwrite(imagename+'-line'+str(m)+'-rot'+str(angle_min)+'.png', linecrop)
        
        #plt.figure(dpi=300)
        #plt.imshow(linecrop)
        #plt.savefig(sys.argv[1]+'line'+str(m+'.png')
        m=m+n
        n=1 
    else:
        n= n+1 # grab another valley if the section is too narrow
        
        
        


# #---------- unused start, but may need to copy-paste from

# # Load the image in grayscale
# filename= 'topanribut.png'
# image = cv.imread(filename, cv.IMREAD_COLOR)
# image=  cv.bitwise_not(image)

# height= image.shape[0]
# width= image.shape[1]
# center = (width/2, height/2) 

# CHANNEL= 2 # red is 'brighter as closer to background"
# #image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# gray= image[:,:,CHANNEL]
# _, thresholded = cv.threshold(gray, 0, 1, cv.THRESH_OTSU) # less smear
# _, gray = cv.threshold(gray, 0, 80, cv.THRESH_OTSU) # less smear
# #_, gray = cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)

# # search the optimum orientation,
# c=[]
# moment_min=1e9
# angles= np.linspace(-8,8,200)
# for i in angles:
#     res= rotmoment2(gray,i)
#     c.append(res)
#     moment_i= (res[0]-res[1])
#     if np.abs(moment_i) < moment_min:
#         moment_min= np.abs(moment_i)
#         angle_min= i
        

# # check for orientation w/ moment
# ax=[]
# ay=[]
# for i in angles:
#     cx, cy= rotmoment(thresholded, i)
#     ax.append(cx)
#     ay.append(cy)
    
# hst=[]
# for i in angles:
#     h= rottrap(thresholded, i)
#     hst.append(h)
    
# # find optimal page orientation, ternary search 
# # page orientation usually wants to MAXIMIZE 
# #  the area under the histogram curve
# #left, right = -pow(PHI,3), pow(PHI,3)  # Define the range to search within
# left, right = -6, 6  # Define the range to search within
# epsilon = 1e-6  # Define the desired precision
# while abs(right - left) > epsilon:
#     mid1 = left + (right - left) / 3
#     mid2 = right - (right - left) / 3
#     if rottrap(thresholded, mid1) < rottrap(thresholded, mid2):
#         left = mid1
#     else:
#         right = mid2
# # found it
# orient = (left + right) / 2
# print(orient)

# M = cv.getRotationMatrix2D(center, orient, 1.0)
# thresholded = cv.warpAffine(thresholded, M, (width,height))

# # calculate histogram
# column_indices = np.arange(width)  # Generate array of column indices
# row_indices= np.arange(height)
# histogram_x = np.sum(thresholded, axis=0)  # Sum along columns to get histogram
# histogram_y = np.sum(thresholded, axis=1)  # Sum along columns to get histogram

# # line segments
# #render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
# #render= cv.cvtColor(render[:,:,CHANNEL], cv.COLOR_GRAY2RGB)
# _, renderbw = cv.threshold(thresholded, 0, 240, cv.THRESH_BINARY)
# render= cv.cvtColor(renderbw, cv.COLOR_GRAY2RGB)
# #plt.figure(dpi=300)
# #plt.imshow(render) 
# plt.plot(histogram_y, row_indices)
# #plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
# plt.xlabel('Row Index')
# plt.ylabel('count')
# plt.show()

# from scipy.ndimage import gaussian_filter1d

# def find_peaks_valleys(hst):
#     peaks = []
#     valleys = []
#     for i in range(1, len(hst) - 1):
#         if hst[i] > hst[i - 1] and hst[i] >= hst[i + 1]:
#             peaks.append(i)
#         elif hst[i] < hst[i - 1] and hst[i] <= hst[i + 1]:
#             valleys.append(i)
#         elif hst[i] < hst[i - 1] and i==len(hst)-2:
#             valleys.append(i)    
#     return peaks, valleys

# # smooth the histogram
# histogram_ys= gaussian_filter1d(histogram_y, pow(PHI,3))
# _,valleys= find_peaks_valleys(histogram_ys)
# #valleys.append(len(histogram))
# valleys.insert(0,0) # append 0 at the beginning

# def average_difference(lst):
#     differences = [lst[i+1] - lst[i] for i in range(len(lst)-1)]
#     avg_diff = sum(differences) / len(differences)
#     return avg_diff

# step= average_difference(valleys) / np.sqrt(PHI)

# # gonna crop-select each lines here
# m=0
# n=1
# while (valleys[m]<=max(valleys) and (m+n)<len(valleys)):
#     top=valleys[m]
#     bot=valleys[m+n]
#     if (bot-top)>step:
#         linecrop= renderbw[top:bot,:]
#         lwidth= linecrop.shape[0]
#         lheight= linecrop.shape[1]
#         lcenter= (int(lwidth/2), int(lheight/2))
        
#         # correct the line orientation
#         # line orientation perhaps wants to MINIMIZE
             
#         plt.figure(dpi=300)
#         plt.imshow(linecrop)
#         m=m+n
#         n=1 
#     else:
#         n= n+1


# #-----------

# # spaces, horizontal
# render= image.copy()
# for i in range(width-1, -1, -1):
#     if (histogram_x[i]<=SLIC_SPACE):
#         for j in range(height):
#             render.itemset((j,i,2), 240)
#     elif  (histogram_x[i]<PHI*SLIC_SPACE):
#         for j in range(height):
#             render.itemset((j,i,2), 80)
    
# render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
# plt.figure(dpi=300)
# plt.imshow(render) 
# # Plot the histogram
# plt.plot(column_indices, histogram_x)
# #plt.title('Histogram of Grayscale Image After Otsu Thresholding Along X-Axis')
# plt.xlabel('Column Index')
# plt.ylabel('Number of Pixels')
# plt.show()

# files=[]
# for i in range(2):  # Loop through 0, 1, 2
#     for j in range(10):  # Loop through 0, 1, 2
#         files.append(f"p{i}{j}")  # Append the formatted string followed by a space
