# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:26:51 2024
@author: Dian Andriana
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import imutils
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
#The inputImg represents the image that will be taken by the blind person, and 
#then convert it grayscale.
inputImg = cv2.imread('topanribut.png')
grayImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
plt.imshow(inputImg, cmap='gray')

#We use a filter to blur out the noise from the image.
gaussianFilter = cv2.GaussianBlur(grayImg, (5,5), 0)
plt.imshow(gaussianFilter, cmap="gray")
     
#binarize and invert the image. 
_, binarizedImg = cv2.threshold(gaussianFilter, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.imshow(binarizedImg, cmap="gray")

binarizedImg[binarizedImg == 0] = 1
binarizedImg[binarizedImg == 255] = 0
plt.imshow(binarizedImg, cmap="gray")

#Erosion using skeletonize
skeletonImg = skeletonize(binarizedImg)
plt.imshow(skeletonImg, cmap="gray")

#plt.show()

#topanribut
height = skeletonImg.shape[0] #50 =height
width = skeletonImg.shape[1]  #353
#start_point=(200,16)
print(skeletonImg.shape[0],skeletonImg.shape[1])
# for i in range(height):
#     for j in range(width):
#         k1 = start_point[1]+i
#         k2 = start_point[0]+j
#         print(start_point[1]+i,start_point[0]+j,skeletonImg[start_point[1]+i,start_point[0]+j])
       
# 198,30 --> 188,34  --> y=mx+c
# step = gradient = -0.88 , c = 192

count = 0
step = -0.58
# k3 = #start_point[0] #x coordinate
# k4 = #start_point[1] #y


c= 192
arr_sum = []
list_space_angle_start = []
list_space_angle_end = []
j= width -1   #memotong sumbu x : x = -c/m untuk y = 0, c = -192
print(j)
print(j)
count_contigu = 0 
before = (-1,-1)
list_split = []
while j < (width) and j> 0 :
    j-=1
    k3 = j
    k4 = 0
    sum = 0 
    count +=1
    while k4 < (height-1)  and k3 >-1: #and count <20:       
        k3 = k3-1
        k4 = int(round(k4-step))        
        try :
            sum = sum + skeletonImg[k4,k3]
            print(k3,k4,skeletonImg[k4,k3],sum)
        except: print()
    arr_sum.append(sum)
    
    if sum==0 : 
        if before[0]==(j+1) or before[0]==(j+2) or before[0]==(j+3) or before[0]==(j+4) :
            count_contigu +=1
            if count_contigu > 1 :  #jika ada beberapa berdampingan
                if  len(list_split)>0 :  #jika sudah ada list berdampingan
                    if list_split[-1][0]!=(j+1) and  list_split[-1][0]!=(j+2) and list_split[-1][0]!=(j+3) and list_split[-1][0]!=(j+4) : #jika elemen list terakhir tidak sama dg j+1
                        list_split.append((j,0)) #add baru
                    elif list_split[-1][0]==(j+1) or list_split[-1][0]==(j+2) or list_split[-1][0]==(j+3) or list_split[-1][0]==(j+4) :    
                        #jika elemen list terakhir sama dg j+1, atau berurutan, atau j+2,j+3,j+4
                        list_split[-1]=(j,0)  #update j yg terakhir , untuk yg berdampingan sela 0,1,2,3,4
                else :
                    list_split.append((j,0))
                print(list_split," lstsplt ")
        else:
            count_contigu =0
        #if count_contigu == 0 :
        before = (j,0)
        
        print(count_contigu," countigu ", before)
        
        print(count,"########",j,k3,k4)
        list_space_angle_start.append((j,0))
        list_space_angle_end.append((k3,k4))
print(arr_sum,count)
print("list split ", list_split)

# Python program to explain cv2.line() method 
 
 
# Window name in which image is displayed
window_name = 'Image'


# Green color in BGR
color = (0, 0, 255)

# Line thickness of 9 px
thickness =1

# Using cv2.line() method
# Draw a diagonal green line with thickness of 9 px





for s in range(len(list_space_angle_start)):
    print(list_space_angle_start[s],list_space_angle_end[s])
    image = cv2.line(inputImg, (list_space_angle_start[s]), (list_space_angle_end[s]), color, thickness)
        
# Displaying the image 
cv2.imwrite("angle.jpg", image) 

#cari batas2 nya dulu

# [rows, columns] 
crop = image[0:50, height:100]   #top left   bottom right
height = crop.shape[0] #50 =height
width = crop.shape[1]  #353




center = (width/2, height/2)
angle = 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(crop, rotation_matrix, (width, height))
cv2.imwrite('rotated_image.jpg', rotated_image)

rotated = imutils.rotate_bound(crop, -45)
cv2.imwrite("45angle.jpg", rotated) 

#mask = np.zeros(rotated.shape[:2], dtype="uint8")  #get partial
#cv2.rectangle(mask, (0, 0), (height, width), 255, -1) #top left   bottom right
#cv2.imshow("Rectangular Mask", mask)
# apply our mask 
# cropped out
#masked = cv2.bitwise_and(rotated, rotated, mask=mask)
#cv2.imwrite("masked.jpg", masked)

back_rotated = imutils.rotate_bound(rotated, 45)
cv2.imwrite("back45angle.jpg", back_rotated) 


