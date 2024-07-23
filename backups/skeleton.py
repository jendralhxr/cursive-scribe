#AUTHOR: Bu Dian
from skimage.morphology import thin, skeletonize,medial_axis
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from PIL import Image
import numpy as np

# Invert the horse image
     #invert(data.horse())
#image = Image.open('geroba-16-cut.jpg')

#selanjutnya proses di c:\python27\chaincode.py
import cv2
#image = cv2.imread("mnist-asf.jpg")
#image = cv2.imread("mnist-asf.jpg")
image = cv2.imread("0001_b.png")#"aksara-kuno-1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', image)

# Convert image in grayscale
gray_im = image #cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#plt.subplot(221)
#plt.title('Grayscale image')
#plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

# Contrast adjusting with gamma correction y = 1.2
gray_correct = np.array(255 * (gray_im / 255) ** 2.2 , dtype='uint8')
#plt.subplot(223)
#plt.title('Gamma Correction y= 1.2')
#plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
cv2.imshow('CorrectGray', gray_correct)
image = gray_correct



image = cv2.bitwise_not(image) #inverse bw


  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(image,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)  #cv2.CHAIN_APPROX_NONE) 



# Grayscale 
#gray =  #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv2.Canny(image, 30, 200) 
cv2.waitKey(0) 
  
cv2.imshow('Canny Edges After Contouring', edged) 
#cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', image)
#mask=image
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 



#mask = cv2.threshold(image,60,255,cv2.THRESH_BINARY) 
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


image = cv2.bitwise_not(image) #inverse bw


ret,mask = cv2.threshold(image,127,255,cv2.THRESH_OTSU) #cv2.THRESH_BINARY) #127
#thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
#thresh = cv2.bitwise_not(thresh)
#plt.show()

    # get the contours in the thresholded image
#(_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)  #cv2.CHAIN_APPROX_NONE) 
print("Again Number of Contours found = " + str(len(contours2))) 

cv2.drawContours(mask, contours2, -1, (0, 255, 0), 3) 

cv2.imshow('newContours2', mask)
ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_OTSU)#cv2.THRESH_BINARY) #127
cv2.imshow('newThres', mask)


mask = cv2.bitwise_not(mask) #inverse bw

#mask = np.zeros(image.shape[:2], dtype="uint8")
#cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
#cv2.imshow("Rectangular Mask", mask)
# apply our mask -- notice how only the person in the image is
# cropped out
# perform skeletonization
skeleton = cv2.ximgproc.thinning(mask, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN) # prefers straight lines

#skeleton = thin(mask) #skeletonize(mask, method='lee') #thin(mask)#medial_axis(mask)  #untuk 0005_l.png
#morphology.skeletonize(mask, method='lee'), morphology.medial_axis(mask), or morphology.thin(mask)

# display results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)


ax[1].set_title('Threshold', fontsize=20)
ax[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
ax[1].axis('off')

ax[2].imshow(skeleton, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('skeleton', fontsize=20)
plt.imsave('skeleton-out.jpg', skeleton, cmap='gray')


fig.tight_layout()



plt.show()

#masked = cv2.bitwise_and(image, image, mask=mask)
#cv2.imshow("Mask Applied to Image", mask)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 

