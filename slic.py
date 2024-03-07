# USAGE: python -u scribe.py IMAGE_INPUT 

import numpy as np
import cv2 as cv
import sys

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
#lic.iterate(cue, num_iterations=4)

slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.MSLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelLSC(cue, region_size = space)
slic.iterate()

mask= slic.getLabelContourMask()
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

moments = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue.item(j,i)!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)

render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
# non-void superpixel
for n in range(num_slic):
    if ( len(moments[n])>1):
        cx= int( np.average(moments[n][:,0]) )
        cy= int( np.average(moments[n][:,1]) )
        render.itemset((cy,cx,1), 255)
        print(f'point{n} at ({cx},{cy})')

# for cls_lbl in range(num_slic):
#     fst_cls = np.argwhere(lbls==cls_lbl)
#     x, y = fst_cls[:, 0], fst_cls[:, 1]
#     c = (x.mean(), y.mean())
#     print(f'{cls_lbl}:{len(c)}')
#     # cx= int(c[1])
#     # cy= int(c[0])
#     # if (cue.item(cy,cx) != 0):
#     #     render.itemset((cy,cx,1), 255)
#     #     print(f'{cls_lbl} point at: ({int(c[1])}, {int(c[0])})')
# #     # else:
# #     #     render.itemset((cy,cx,2), 255)
# #     #     print(f'{cls_lbl} void at: ({int(c[1])}, {int(c[0])})')
    
render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
render= cv.bitwise_or(render, mask2)
cv.imwrite(sys.argv[2], render)
print(f'save to: {sys.argv[2]}')
#cv.imshow("mask", mask)
cv.imshow("show", render)
key = cv.waitKey(0) & 0xff

