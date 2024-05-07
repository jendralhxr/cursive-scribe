# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:39:58 2024

@author: rdx
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
os.chdir("D:")

SLIC_SPACE= 3
PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

filename= "topanribut.png"
image = cv.imread(filename)
image=  cv.bitwise_not(image)

CHANNEL= 2
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, 200, cv.THRESH_OTSU) # less smear
#_, gray= cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)


# ORB
orb = cv.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
image_with_keypoints = cv.drawKeypoints(gray, keypoints, None)

# SIFT
sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
image_with_keypoints = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


