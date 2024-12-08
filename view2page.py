import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)
SLIC_SPACE= 8
k=7
THREVAL= 60

def draw(img): # draw the bitmap
    plt.figure(dpi=600)
    plt.grid(False)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))

filename=sys.argv[1]
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
#image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

image_gray= image[:,:,2]
_, image_binary = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU) # less smear
column_indices = np.arange(width)  
row_indices = np.arange(height)

histogram_x = np.sum(image_binary, axis=0)/THREVAL  # Sum along columns
diff_abs = np.abs ( np.diff(histogram_x))

edges = np.where(diff_abs > height/pow(PHI,k+1))[0]
# image_bgr= image.copy()
# for x in valleys1[0]:
#     cv.line(image_bgr, (x, 0), (x, image_bgr.shape[0]), (0, 0, 255), 8)  # Red line
# for x in edges:
#     cv.line(image_bgr, (x, 0), (x, image_bgr.shape[0]), (0, 255, 0), 8)  # green line
# draw(image_bgr)

threshold = pow(PHI,2)*SLIC_SPACE  # Adjust based on expected cluster proximity
clusters = [[edges[0]]]

for value in edges[1:]:
    if value - clusters[-1][-1] <= threshold:
        clusters[-1].append(value)
    else:
        clusters.append([value])
cluster_centers_x = [np.mean(cluster) for cluster in clusters]
[np.mean(cluster) for cluster in clusters]

for n in range(1, len(cluster_centers_x)):
    pagecrop= image_binary[:,int(cluster_centers_x[-(n+1)]):int(cluster_centers_x[-n])]
    histogram_y = np.sum(pagecrop, axis=1)/THREVAL  # Sum along rows
    diff_abs = np.abs ( np.diff(histogram_y))
    edges = np.where(diff_abs > height/pow(PHI,k))[0]
    clusters = [[edges[0]]]
    for value in edges[1:]:
        if value - clusters[-1][-1] <= threshold*PHI:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    cluster_centers_y = [np.mean(cluster) for cluster in clusters] # ideally only two values
    pagecrop= image[int(cluster_centers_y[0]):int(cluster_centers_y[1]),\
                           int(cluster_centers_x[-(n+1)]):int(cluster_centers_x[-n])]
    
    cv.imwrite(imagename+'v'+str(n)+'.png', pagecrop)
        