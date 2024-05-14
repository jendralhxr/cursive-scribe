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
import networkx as nx

plt.figure(dpi=300)

def draw1(img): # draw the intensity
    plt.figure(dpi=300)
    plt.imshow(img)
  
def draw2(img): # draw the bitmap
    plt.figure(dpi=300)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        
SLIC_SPACE= 3
PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

filename= "topanribut.png"
image = cv.imread(filename)
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]


CHANNEL= 2
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, 80, cv.THRESH_OTSU) # less smear
#_, gray= cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)


# ORB
orb = cv.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
image_with_keypoints = cv.drawKeypoints(gray, keypoints, None)

def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp= cv.filter2D(img, -1, kernel)
    return(sharp)

sharp1= sharpen(gray)
sharp2= sharpen(sharp1)

# SIFT
sift = cv.SIFT_create()
kp0, descriptors = sift.detectAndCompute(gray, None)
ks0 = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp1, descriptors = sift.detectAndCompute(sharp1, None)
ks1 = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp2, descriptors = sift.detectAndCompute(sharp2, None)
ks2 = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cue= gray.copy()
render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

#SLIC
slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = SLIC_SPACE)
slic.iterate()
mask= slic.getLabelContourMask()
result_mask = cv.bitwise_and(cue, mask)
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

# moments calculation for each superpixels, either voids or filled (in-stroke)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
moments_void = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue.item(j,i)!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render.itemset((j,i,0), 120-(10*(lbls[j,i]%6)))
        else:
            moments_void[lbls[j,i]] = np.append(moments_void[lbls[j,i]], np.array([[i,j]]), axis=0)

# generating nodes
scribe= nx.Graph() # start anew, just in case

# valid superpixel
filled=0
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE): # remove spurious superpixel with area less than 2 px 
        cx= int( np.mean(moments[n][1:,0]) ) # centroid
        cy= int( np.mean(moments[n][1:,1]) )
        if (cue.item(cy,cx)!=0):
            render.itemset((cy,cx,1), 255) 
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), pos_bitmap=(cx,cy), pos_render=(cx,-cy) )
            #print(f'point{n} at ({cx},{cy})')
            filled=filled+1

# connected components
from dataclasses import dataclass, field
from typing import List

@dataclass
class ConnectedComponents:
    bgn_x: int
    bgn_y: int
    end_x: int
    end_y: int
    mid_x: int
    mid_y: int
    nodes: List[int] = field(default_factory=list)

# component = ConnectedComponents(bgn_x=0, bgn_y=0, end_x=100, end_y=100, mid_x=50, mid_y=50, nodes=[2])

pos = nx.get_node_attributes(scribe,'pos_bitmap')
for n in range(scribe.number_of_nodes()):
    # fill
    seed= pos[n]
    ccv= gray.copy()
    cv.floodFill(ccv, None, seed, 200, loDiff=(5), upDiff=(5))
    _, ccv = cv.threshold(ccv, 100, 200, cv.THRESH_BINARY)
    #draw2(ccv)
    cv.imwrite(str(n)+'-'+str(pos[n][0])+'-'+str(pos[n][1])+'.png', ccv)
    # check
    # append
    
        