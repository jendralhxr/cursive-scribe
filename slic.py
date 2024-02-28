# USAGE: python -u scribe.py IMAGE_INPUT 

import numpy as np
import cv2 as cv
import sys
from scipy.signal import find_peaks

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]
image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)
ret, image_thr = cv.threshold(image_gray, 0, 240, cv.THRESH_OTSU) # other thresholding method may also work

space= 16
key= 32;

while (key!=27 and key!=ord('q') ):

    cv_slic = cv.ximgproc.createSuperpixelSLIC(image_thr,algorithm = cv.ximgproc.SLICO, region_size = space)
    cv_slic.iterate()
    mask= cv_slic.getLabelContourMask()
    render = cv.cvtColor(image_thr, cv.COLOR_GRAY2BGR)
    render = 
    render[:,:,1]= mask

    cv.imshow("show", render)
    #cv.imshow("mask", mask)
    if key==ord('a'):
        space= space+1
        print(space)
    if key==ord('d'):
        space= space-1
        print(space)
    if key==ord('s'):
        cv.imwrite(str(space)+".png", render)
        print("save")
     
    key = cv.waitKey(1) & 0xff
