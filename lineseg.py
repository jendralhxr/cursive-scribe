# USAGE: python -u scribe.py IMAGE_INPUT IMAGE_OUTPUT

import numpy as np
import cv2 as cv
import sys
from scipy.signal import find_peaks

BAND=300    
SKEW_MAX= 20

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]

key=0


PROMINENCE= 80

while (key!=27 and key!=ord('q') ):
    image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)
    #ret, image_thresh1 = cv.threshold(image_gray, 0, 240, cv.THRESH_TRIANGLE) # other thresholding method may also work
    ret, image_thr = cv.threshold(image_gray, 0, 240, cv.THRESH_OTSU) # other thresholding method may also work
    width=image_gray.shape[1]
    height=image_gray.shape[0]

    band_left = np.zeros(height, dtype=np.uint16)
    band_right = np.zeros(height, dtype=np.uint16)
    
    current= image_thr.copy()
    peaks_left= []
    peaks_right= []
    
    for j in range(height):
        for i in range(BAND):
            if current.item(j,width-BAND-i):
                band_left[j]= band_left[j]+1
            if current.item(j,width-1-i):
                band_right[j]= band_right[j]+1
        #print("{}:{} -- {}".format(j, band_left[j], band_right[j]))

    peaks_left, _ = find_peaks(band_left, PROMINENCE)
    peaks_right, _ = find_peaks(band_right, PROMINENCE)
    #print("right", peaks_right)
    #print("left", peaks_left)
    
    cue= (cv.cvtColor(current, cv.COLOR_GRAY2BGR))
    for j in peaks_right:
        #for i in peaks_left:
        #    if ((i-j)<SKEW_MAX) and ((i-j)>=0):
                #print()
        #print(j,i)
        cv.line(cue, (0,j), (width-1,j), (0,255,0), 1) 

    if key==ord('a'):
        PROMINENCE = PROMINENCE -1
        print("prom: "+str(PROMINENCE))
    if key==ord('d'):
        PROMINENCE = PROMINENCE +1
        print("prom: "+str(PROMINENCE))
    if key==ord('s'):
        cv.imwrite('cueline.png', cue)
        print("save")
     
    cv.imshow('cue', cue)
    key = cv.waitKey(0) & 0xff
