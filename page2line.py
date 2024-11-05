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

def average_distance_between_peaks(peaks):
    if len(peaks) < 2:
        return 0  # Not enough peaks to calculate a distance
    
    # Calculate distances between consecutive peaks
    distances = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]
    
    # Calculate the average distance
    average_distance = sum(distances) / len(distances)
    return average_distance

# Example usage
# hst = [0, 3, 1, 4, 2, 5, 2, 1, 3, 1]  # Replace with your histogram data
# peaks, valleys = find_peaks_valleys(hst)
# print("Peaks:", peaks)
# print("Valleys:", valleys)
from scipy.signal import find_peaks

def draw_histograms_on_image(thresholded_image, histogram_x, histogram_y):
    # Normalize histograms to fit within the image dimensions for visualisation
    histogram_x_norm = (histogram_x / histogram_x.max()) * thresholded_image.shape[0] / PHI
    histogram_y_norm = (histogram_y / histogram_y.max()) * thresholded_image.shape[1] / PHI
    
    plt.imshow(thresholded_image, cmap='gray')
    # Overlay histogram_y as green scatter points (along the left, vertically)
    y_points = np.arange(len(histogram_y_norm))
    x_points = histogram_y_norm  # Align to start from the left of the image
    plt.plot(x_points, y_points, color='green', linewidth=1)  # Connect points with a line
    # Overlay histogram_x as red scatter points (along the top, horizontally)
    # x_points = np.arange(len(histogram_x_norm))
    # y_points = thresholded_image.shape[0] - histogram_x_norm  # Adjust to align at top of image
    # plt.plot(x_points, y_points, color='red', linewidth=1)  # Connect points with a line
    
    # Find peaks and valleys in histogram_y
    peaks = find_peaks(histogram_y)[0]
    valleys = find_peaks(thresholded_image.shape[1]-histogram_y)[0]

    # # Draw red horizontal lines at each peak of histogram_y
    # for peak in peaks:
    #     plt.axhline(y=peak, color='red', linestyle='--', linewidth=0.5)
    # Draw blue horizontal lines at each valley of histogram_y
    for valley in valleys:
        plt.axhline(y=valley, color='blue', linestyle='--', linewidth=0.5)


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

# rect_out=[]
# rect_fit=[]
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

def find_optimal_Z(thresholded_image, Z_min=0, Z_max=10, Z_step=1):
    best_Z = Z_min
    max_avg_distance = 0
    
    for Z in np.arange(Z_min, Z_max, Z_step):
        # Apply Gaussian filter with sigma = PHI^Z
        histogram_y = gaussian_filter1d(np.sum(thresholded_image, axis=1), pow(PHI, Z))
        
        # Find peaks and valleys
        peaks = find_peaks(histogram_y)[0]
        
        # Calculate average distance between peaks
        avg_distance = average_distance_between_peaks(peaks)
        
        # Check if this is the best Z found so far
        if avg_distance > max_avg_distance:
            max_avg_distance = avg_distance
            best_Z = Z

    return best_Z, max_avg_distance

# smooth the histogram
Z, interval= find_optimal_Z(thresholded)
histogram_x =gaussian_filter1d( np.sum(thresholded, axis=0), pow(PHI,Z))  # Sum along columns
histogram_y =gaussian_filter1d( np.sum(thresholded, axis=1), pow(PHI,Z))  # Sum along rows
draw_histograms_on_image(gray, histogram_x, histogram_y)

# find the line segment
peaks= find_peaks(histogram_y)[0]
valleys= find_peaks(thresholded.shape[1]-histogram_y)[0]
valleys = np.insert(valleys, 0, 0) # append zero as the first valley

def average_difference(lst):
    differences = [lst[i+1] - lst[i] for i in range(len(lst)-1)]
    avg_diff = sum(differences) / len(differences)
    return avg_diff


image=  cv.bitwise_not(image)

step= average_difference(valleys) * PHI

# gonna crop-select each lines here
m=0
n=1
while (valleys[m]<=max(valleys) and (m+n)<len(valleys)):
    top=valleys[m]
    bot=valleys[m+n]
    if (bot-top)>step:
        linecrop= gray[top:bot,:]
        linecrop_img= image[top:bot,:]
        
        # # find the line orientation
        # c=[]
        # moment_min=1e9
        # angles= np.linspace(-6,6,200)
        # for i in angles:
        #     res= rotmoment2(linecrop,i)
        #     c.append(res)
        #     moment_i= (res[0]-res[1])
        #     if np.abs(moment_i) < moment_min:
        #         moment_min= np.abs(moment_i)
        #         angle_min= i     
        # if angle_min>-3 and angle_min<3:
        #     linecrop= rotimg(linecrop, angle_min)
        #     linecrop_img= rotimg(linecrop_img, angle_min)
        # print(f"{imagename}: line {m}, angle {angle_min} ")                
        
        #cv.imwrite(imagename+'-line'+str(m)+'.png', linecrop)
        cv.imwrite(imagename+'-lineimg'+str(m)+'.png', linecrop_img)
        
        m=m+n
        n=1 
    else:
        n= n+1 # grab another valley if the section is too narrow
        
        
