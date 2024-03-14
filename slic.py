# USAGE: python -u scribe.py IMAGE_INPUT 

import numpy as np
import cv2 as cv
import sys
import networkx as nx
import matplotlib.pyplot as plt


# freeman code going anti-clockwise like trigonometrics angle
"""
3   2   1
  \ | /
4 ------0
  / | \
5   6   7
"""
phi= 1.6180339887498948482

def freeman(x, y):
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (y==0):
        y=1e-9
    if (abs(x/y)<phi) and (abs(y/x)<phi): # corner angles
        if   (x>0) and (y>0):
            return(1)
        elif (x<0) and (y>0):
            return(3)
        elif (x<0) and (y<0):
            return(5)
        elif (x>0) and (y<0):
            return(7)
    else: # square angles
        if   (x>0) and (abs(x)>abs(y)):
            return(int(0))
        elif (y>0) and (abs(y)>abs(x)):
            return(2)
        elif (x<0) and (abs(x)>abs(y)):
            return(4)
        elif (y<0) and (abs(y)>abs(x)):
            return(6)
        

######## main routine

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]
image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)

ret, cue = cv.threshold(image_gray, 0, 120, cv.THRESH_OTSU) # other thresholding method may also work
render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)

# LSC and SLIC 
space= 3
key= 32;

# SEEDS parameters
num_superpixels = 5000
num_levels = 4
prior = 2
num_histogram_bins = 5
double_step = False

#slic = cv.ximgproc.createSuperpixelSEEDS(cue.shape[1], cue.shape[0], 1, num_superpixels, num_levels, prior, num_histogram_bins, double_step)
#slic.iterate(cue, num_iterations=4)

slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.MSLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelLSC(cue, region_size = space)
slic.iterate()

#mask= slic.getLabelContourMask()
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue.item(j,i)!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render.itemset((j,i,0), 120-(10*(lbls[j,i]%6)))
            

isi=0
# non-void superpixel
for n in range(num_slic):
    if ( len(moments[n])>1):
        cx= int(moments[n][:,0][1])
        cy= int(moments[n][:,1][1])
        render.itemset((cy,cx,0), 255)
        cx= int(moments[n][:,0][-1])
        cy= int(moments[n][:,1][-1])
        render.itemset((cy,cx,1), 255)
        cx= int( np.mean(moments[n][1:,0]) )
        cy= int( np.mean(moments[n][1:,1]) )
        render.itemset((cy,cx,2), 255)
        #print(f'point{n} at ({cx},{cy})')
        isi= isi+1

#print(isi)
#cv.imshow("show", render)
#key = cv.waitKey(0) & 0xff
    
#render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
#mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
#render= cv.bitwise_or(render, mask2)
cv.imwrite(sys.argv[2], render)
print(f'save to: {sys.argv[2]}')
#cv.imshow("mask", mask)
#cv.imshow("show", render)
#key = cv.waitKey(0) & 0xff

