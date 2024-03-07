# USAGE: python -u scribe.py IMAGE_INPUT 

import numpy as np
import cv2 as cv
import sys

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]
image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)

ret, image_thr = cv.threshold(image_gray, 0, 120, cv.THRESH_OTSU) # other thresholding method may also work
render = cv.cvtColor(image_thr, cv.COLOR_GRAY2BGR)

space= 3
key= 32;

# Define SEEDS algorithm parameters
num_superpixels = 5000
num_levels = 4
prior = 2
num_histogram_bins = 5
double_step = False

slic = cv.ximgproc.createSuperpixelSEEDS(image_thr.shape[1], image_thr.shape[0], 1, num_superpixels, num_levels, prior, num_histogram_bins, double_step)
slic.iterate(image_thr, num_iterations=4)

#slic = cv.ximgproc.createSuperpixelSLIC(image_thr,algorithm = cv.ximgproc.SLICO, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(image_thr,algorithm = cv.ximgproc.SLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(image_thr,algorithm = cv.ximgproc.MSLIC, region_size = space)

#slic = cv.ximgproc.createSuperpixelLSC(image_thr, region_size = space)



#slic.iterate()
mask= slic.getLabelContourMask()
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

print(num_slic)
render = cv.cvtColor(image_thr, cv.COLOR_GRAY2BGR)

for cls_lbl in range(num_slic):
    fst_cls = np.argwhere(lbls==cls_lbl)
    x, y = fst_cls[:, 0], fst_cls[:, 1]
    c = (x.mean(), y.mean())
    print(f'{lbls}:{c}')
    # cx= int(c[1])
    # cy= int(c[0])
    # if (image_thr.item(cy,cx) != 0):
    #     render.itemset((cy,cx,1), 255)
    #     print(f'{cls_lbl} point at: ({int(c[1])}, {int(c[0])})')
#     # else:
#     #     render.itemset((cy,cx,2), 255)
#     #     print(f'{cls_lbl} void at: ({int(c[1])}, {int(c[0])})')
    
render = cv.cvtColor(image_thr, cv.COLOR_GRAY2BGR)
mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
render= cv.bitwise_or(render, mask2)
cv.imshow("show", render)
cv.imwrite(sys.argv[2], render)
print(f'save to: {sys.argv[2]}')
#cv.imshow("mask", mask)
key = cv.waitKey(0) & 0xff

