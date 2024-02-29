# USAGE: python -u scribe.py IMAGE_INPUT IMAGE_OUTPUT

import numpy as np
import cv2 as cv
import sys

# heatmap
COLOR_MIN = 30
COLOR_MAX = 230
COLOR_RANGE= 200

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]

image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)
ret, image_thresh = cv.threshold(image_gray, 0, 240, cv.THRESH_OTSU) # other thresholding method may also work
image_thin = cv.ximgproc.thinning(image_thresh, thinningType = cv.ximgproc.THINNING_ZHANGSUEN) # prefers straight lines
#image_thin = cv.ximgproc.thinning(image_thresh, thinningType = cv.ximgproc.THINNING_GUOHALL)  # prefers squiggly lines

#contours, hierarchy = cv.findContours(image_thin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # 200 for 0000_a
#contours, hierarchy = cv.findContours(image_thin, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1) # 103 for 0000_a
contours, hierarchy = cv.findContours(image_thin, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS) # 95 for 0000_a
#print("{}: {} contours".format(filename, len(contours)))
#print(contours)
#arr= np.array(contours)
#print("{}: {} points".format(filename, arr.shape))
#print("{}: {} points".format(filename, arr.shape[1]))

#someoverlay
#overlay = np.zeros([height, width, 3], dtype=np.uint8)
#contour_length= arr.shape[1]
#for i in range(contour_length):
#    x= arr[0,i,0,0];
#    y= arr[0,i,0,1];
#    cv.circle(overlay, (x,y), 2, (0,int(COLOR_MAX - i*COLOR_RANGE/contour_length),0), -1)

cv.imshow('a', image_gray)
cv.imshow('b', image_thresh)
cv.imshow('c', image_thin)
cv.imwrite('1.png', image_gray)
cv.imwrite('2.png', image_thresh)
cv.imwrite('3.png', image_thin)
#render= cv.cvtColor(image_thin, cv.COLOR_GRAY2BGR)
#render= cv.bitwise_or(render, image)
#render= cv.bitwise_or(render, overlay)
#cv.imwrite(sys.argv[2], render)
#cv.imshow('c', render)

key = cv.waitKey(0) & 0xff
if key==27:
    quit()

