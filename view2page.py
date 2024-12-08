import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)


def draw(img): # draw the bitmap
    plt.figure(dpi=600)
    plt.grid(False)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))


filename='p01.jpg'
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
#image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

THREVAL= 60

image_gray= image[:,:,2]
_, image_binary = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU) # less smear

column_indices = np.arange(width)  
row_indices = np.arange(height)
histogram_x = np.sum(image_binary, axis=0)/THREVAL  # Sum along columns

from scipy.signal import find_peaks
valleys= find_peaks(-histogram_x, threshold=20)

image_bgr= image.copy()
for x in valleys[0]:
    cv.line(image_bgr, (x, 0), (x, image_bgr.shape[0]), (0, 0, 255), 1)  # Red line
# valley masih kurang okeh

histogram_y = np.sum(image_binary, axis=0)/THREVAL  # Sum along rows
