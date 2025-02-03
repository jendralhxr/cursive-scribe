# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:26:51 2024

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import imutils

# from tensorflow.keras.models import load_model
# from PIL import Image, ImageOps
# The inputImg represents the image
# then convert it grayscale.
# hasil crp-1,2,3,....png hasil potongan angle histogram
inputImg = cv2.imread(
    "dengarkan.png"
)  # PerangJohor-4.png') #p01-lineimg0_n0002_label29.png') #'topanribut.png')  ganti yg lain
# coba data training p01-lineimg0_n0003_label01.png potongan angle di pinggir
input2 = cv2.imread(
    "dengarkan.png"
)  # PerangJohor-4.png') #p01-lineimg0_n0002_label29.png') #'topanribut.png')
grayImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)  # grayImg
plt.imshow(inputImg, cmap="gray")  # grayImg


# PerangJohor-4.png mungkin sudut bukan 45 degree

# We use a filter to blur out the noise from the image.
gaussianFilter = cv2.GaussianBlur(grayImg, (5, 5), 0)  # grayImg,(5,5)
plt.imshow(gaussianFilter, cmap="gray")
cv2.imwrite("GaussianFilter.png", gaussianFilter)

# binarize and invert the image.  # gaussianFilter
# _, binarizedImg = cv2.threshold(gaussianFilter, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, binarizedImg = cv2.threshold(
    gaussianFilter, 0, 180, cv2.THRESH_BINARY | cv2.THRESH_OTSU
)


plt.imshow(binarizedImg, cmap="gray")

binarizedImg[binarizedImg < 180] = 0  # 1
binarizedImg[binarizedImg > 180] = 1  # 0

# binarizedImg[binarizedImg == 0] = 1  #1  untuk topanribut
# binarizedImg[binarizedImg == 255] = 0  #0 untuk topanribut
plt.imshow(binarizedImg, cmap="gray")
cv2.imwrite("binarizedImg.png", binarizedImg)

# #Erosion using skeletonize
skeletonImg = skeletonize(binarizedImg)
plt.imshow(skeletonImg, cmap="gray")
# cv2.imwrite("skeletonImg.png", skeletonImg)
plt.show()

# topanribut
height = skeletonImg.shape[0]  # 50 =height  #skeletonImg
width = skeletonImg.shape[1]  # 353   #skeletonImg
# start_point=(200,16)
# print(binarizedImg.shape[0],skeletonImg.shape[1])
# for i in range(height):
#     for j in range(width):
#         k1 = start_point[1]+i
#         k2 = start_point[0]+j
#         print(start_point[1]+i,start_point[0]+j,skeletonImg[start_point[1]+i,start_point[0]+j])

# 198,30 --> 188,34  --> y=mx+c
# step = gradient = -0.88 , c = 192

count = 0
step = -0.48  # -0.1  #-0.58

# k3 = #start_point[0] #x coordinate
# k4 = #start_point[1] #y


c = 192
# arr_sum = []
list_space_angle_start = []
list_space_angle_end = []
j = width - 1  # memotong sumbu x : x = -c/m untuk y = 0, c = -192
print("width ", width)
count_contigu = 0
before = (-1, -1)
list_split = []
while j < (width) and j > 0:
    j -= 1
    k3 = j
    k4 = 0
    sum = 0
    count += 1
    while k4 < (height - 1) and k3 > -1:  # and count <20:
        k3 = round(k3 + (1 / step))  # k3-1
        k4 = k4 + 1  # int(round(k4-step))
        # print("saja k3,k4 ",k3,k4)
        try:
            sum = sum + skeletonImg[k4, k3]  # skeletonImg
            # print("k3,k4 ",k3,k4,skeletonImg[k4,k3],sum) #skeletonImg
        except:
            print()
    # arr_sum.append(sum)

    if sum == 0:
        if (
            before[0] == (j + 1)
            or before[0] == (j + 2)
            or before[0] == (j + 3)
            or before[0] == (j + 4)
        ):
            count_contigu += 1
            if count_contigu > 0:  # jika ada beberapa berdampingan, dipilih yg contigu
                if len(list_split) > 0:  # jika sudah ada list berdampingan
                    if (
                        list_split[-1][0] != (j + 1)
                        and list_split[-1][0] != (j + 2)
                        and list_split[-1][0] != (j + 3)
                        and list_split[-1][0] != (j + 4)
                    ):  # jika elemen list terakhir tidak sama dg j+1
                        list_split.append((j, 0))  # add baru
                        list_space_angle_start.append((j, 0))
                        list_space_angle_end.append((k3, k4))
                    elif (
                        list_split[-1][0] == (j + 1)
                        or list_split[-1][0] == (j + 2)
                        or list_split[-1][0] == (j + 3)
                        or list_split[-1][0] == (j + 4)
                    ):
                        # jika elemen list terakhir sama dg j+1, atau berurutan, atau j+2,j+3,j+4
                        list_split[-1] = (
                            j,
                            0,
                        )  # update j yg terakhir , untuk yg berdampingan sela 0,1,2,3,4
                        list_space_angle_start[-1] = (j, 0)
                        list_space_angle_end[-1] = (k3, k4)
                else:
                    list_split.append((j, 0))
                    list_space_angle_start.append((j, 0))
                    list_space_angle_end.append((k3, k4))
                print(list_split, " lstsplt ")
        else:
            count_contigu = 0
        # if count_contigu == 0 :
        before = (j, 0)

        print(count_contigu, " countigu ", before)

        print(count, "########", j, k3, k4)
        # tidak dipilih

# print(arr_sum,count)
print("list split ", list_split)
print("len list_split", len(list_split))


# deg =-ATAN((y-c)/x)*180

# Python program to explain cv2.line() method


# Window name in which image is displayed
# window_name = 'Image'


# Green color in BGR
color = (0, 0, 255)

# Line thickness of 9 px
thickness = 1

# Using cv2.line() method
# Draw a diagonal green line with thickness of 9 px


for s in range(len(list_space_angle_start)):
    print("list start ", list_space_angle_start[s], list_space_angle_end[s])
    image = cv2.line(
        inputImg,
        (list_space_angle_start[s]),
        (list_space_angle_end[s]),
        color,
        thickness,
    )

if len(list_space_angle_start) > 0:
    dy = list_space_angle_end[0][0] - list_space_angle_start[0][0]
    dx = list_space_angle_end[0][1] - list_space_angle_start[0][1]
    deg = -1 * int(np.rad2deg(np.arctan((dy) / dx)))

print("deg ", deg)

# Displaying the image
cv2.imwrite("angle.jpg", image)

center = (width / 2, height / 2)  # (0, 0)


print("dy ", dy, list_space_angle_end[0][0], list_space_angle_start[0][0])
print("dx ", dx, list_space_angle_end[0][1], list_space_angle_start[0][1])

# U = [[],[]]
U = [
    [list_space_angle_start[0][0], list_space_angle_start[0][1]],
    [list_space_angle_end[0][0], list_space_angle_end[0][1]],
]


V = [[252, 22], [252, 299]]  # by sample

T = np.matmul(V, np.linalg.inv(U))
print("U ", U)
print("T ", T)


# cari batas2 nya dulu

# [rows, columns]
# crop = image[0:50, height:100]   #top left   bottom right

# top left ambil dari list split sebagai titik awal kemudian dirotasi


# height = crop.shape[0] #50 =height
# width = crop.shape[1]  #353


res = np.asarray(
    list_split
)  # list hasil sbelum rotate  #ralat:BELUM, res dari list_start dan list_end
print("res ", res)  # start points of red lines

res_start = np.asarray(list_space_angle_start)
res_end = np.asarray(list_space_angle_end)

angle = deg  # 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
print("rotation_matrix ", rotation_matrix)
print("center ", center)

resmul = np.matmul(
    res, rotation_matrix
)  # multiply for rotation transform for the start points of red lines
resmul_start = np.matmul(res_start, rotation_matrix)  # rotated
resmul_end = np.matmul(res_end, rotation_matrix)  # rotated

print("resmul ", (resmul))  # begin of lines

print("resmul 01", (int(resmul[0][0]), int(resmul[0][1])))

end_resmul = np.copy(resmul)  # end of lines

bott_right = np.copy(end_resmul)

# resmul[0][0] = resmul[0][0] +  rotation_matrix[0][0] * height
# resmul[1][0] = resmul[1][0] +  rotation_matrix[0][0] * height

# end_resmul[0][0] = end_resmul[0][0] -  rotation_matrix[0][0] * height
# end_resmul[1][0] = end_resmul[1][0] -  rotation_matrix[0][0] * height

# #transform
# print("len resmul",len(resmul))   #for 45 degreee
# for r in range (len(resmul)) :   #for 45 degreee
#     print("r ",r)
#     end_resmul[r][0] = end_resmul[r][0] -  rotation_matrix[0][0] * height
#     if r>0 :
#         bott_right[r][0] = end_resmul[r-1][0]
#         bott_right[r][1] = end_resmul[r-1][1]


#     else :
#         bott_right[r][0] = width - rotation_matrix[0][0] * height
#         #bott_right[0][1] = end_resmul[0][1] #bagaimana nilai y nya

#     resmul[r][0]     = resmul[r][0] +  rotation_matrix[0][0] * height

# print("add resmul 01",(int(resmul[0][0]),int(resmul[0][1])))  #posisi = 229,20 = 229,(284-264)= (resmul[0][1], ((putdraw_rotated.width-resmul[0][0]) )
# print("add resmul 11",(int(resmul[1][0]),int(resmul[1][1])))  #posisi = 218,31=  218,284-253  = (resmul[1][1], ((putdraw_rotated.width-resmul[1][0]) )

resmul3 = np.matmul(res, rotation_matrix)  # T)

T_trans = np.asarray(T)
print("resmul3 ", resmul3)


rotated_image = cv2.warpAffine(
    image, rotation_matrix, (width, height)
)  #  dari list  split        #crop, rotation_matrix, (width, height))
# rotated_image2 = cv2.warpAffine(input2, rotation_matrix, (width, height))     #  dari list  split        #crop, rotation_matrix, (width, height))
cv2.imwrite("rotated_image.jpg", rotated_image)
# cv2.imwrite('rotated_image2.jpg', rotated_image2)

# rotated_image3 = cv2.warpAffine(image, T, (width, height))     #  dari list  split        #crop, rotation_matrix, (width, height))

# cv2.imwrite('3rotated_image.jpg', rotated_image3)
# print("rotated list split ", rotated_image)


rotated = imutils.rotate_bound(image, -deg)  # 45
rotated2 = imutils.rotate_bound(input2, -deg)
cv2.imwrite(str(int(deg)) + "angle.jpg", rotated)

rot_pos = []
rot_pos_end = []

for r in range(len(list_space_angle_start)):
    x3_end = (list_space_angle_end[r][0] - center[0]) * np.cos(deg / 180 * np.pi)

    x3 = (list_space_angle_start[r][0] - center[0]) * np.cos(deg / 180 * np.pi)

    x4_end = -1 * (list_space_angle_end[r][1] - center[1])

    x4 = -1 * (list_space_angle_start[r][1] - center[1])

    x5_end = x4_end * np.sin(deg / 180 * np.pi)
    x5 = x4 * np.sin(deg / 180 * np.pi)

    x6_end = x3_end - x5_end
    x6 = x3 - x5
    # print("x3=list_space_angle_start[r][0] - center[0]) * np.sin(deg/180*np.pi)",list_space_angle_start[r][0], center[0], np.sin(deg/180*np.pi))
    x33 = (list_space_angle_start[r][0] - center[0]) * np.sin(deg / 180 * np.pi)

    x33_end = (list_space_angle_end[r][0] - center[0]) * np.sin(deg / 180 * np.pi)
    print(
        "x3_end =list_space_angle_end[r][0] - center[0]) * np.sin(deg/180*np.pi)",
        list_space_angle_end[r][0],
        center[0],
        np.sin(deg / 180 * np.pi),
    )

    x55 = -1 * (list_space_angle_start[r][1] - center[1]) * np.cos(deg / 180 * np.pi)
    x55_end = -1 * (list_space_angle_end[r][1] - center[1]) * np.cos(deg / 180 * np.pi)

    print(
        "x55_end=list_space_angle_end[r][1] + center[1]) * np.cos(deg/180*np.pi)",
        list_space_angle_end[r][1],
        center[1],
        np.sin(deg / 180 * np.pi),
    )
    # * np.sin(deg/180*np.pi)
    y6_end = x33_end + x55_end
    y6 = x33 + x55

    x7_end = x6_end + 0.5 * rotated.shape[1]
    x7 = x6 + 0.5 * rotated.shape[1]

    y7_end = 0.5 * rotated.shape[0] - y6_end
    y7 = 0.5 * rotated.shape[0] - y6

    # x7 = x6 + 0.5 * rotated.shape[1]
    # y7 = 0.5 * rotated.shape[0] - y6

    print("x6_end ", x6_end, r)
    print("x7 end ", x7_end, r)
    print("y6_end ", y6_end, r)
    print("y7 end ", y7_end, r)
    print("x3_end ", x3_end)
    print("x4_end ", x4_end)
    print("x33_end ", x33_end)
    print("x4_end ", x4_end)
    print("x5_end ", x5_end)
    # print("x44_end ",x44_end,r)
    print("x55_end ", x55_end, r)

    rot_pos.append((x7, y7))  # kumpulan titik hasil rotasi
    rot_pos_end.append((x7_end, y7_end))  # kumpulan titik hasil rotasi
    # rot_y.append(y7)

    putdraw = cv2.circle(rotated, (int(x7), int(y7)), 2, (255, 255, 255), 2)
    putdraw = cv2.circle(putdraw, (int(x7_end), int(y7_end)), 2, (0, 255, 255), 2)


print("rotated.shape ", rotated.shape[0], rotated.shape[1])  # 572,374
cv2.imwrite("putdraw.jpg", putdraw)
#


# resmul_image = cv2.warpAffine(res, rotation_matrix, (width, height))
# print("resmul_image ", resmul_image)


# putdraw = cv2.circle(rotated, (int(resmul[0][0]),int(resmul[0][1])), 2, (255,255,255), 2)
# putdraw = cv2.circle(rotated, (int(resmul[1][0]),int(resmul[1][1])), 2, (255,255,255), 2)
# putdraw = cv2.circle(rotated, (int(end_resmul[0][0]),int(resmul[0][1])), 2, (255,255,255), 2)
# putdraw = cv2.circle(rotated, (int(end_resmul[1][0]),int(resmul[1][1])), 2, (255,255,255), 2)

# for r in range (len(resmul)) :
# putdraw = cv2.circle(rotated, (int(resmul[r][0]),int(resmul[r][1])), 2, (255,255,255), 2)
# putdraw = cv2.circle(rotated, (int(end_resmul[r][0]),int(end_resmul[r][1])), 2, (0,255,0), 2)
# if r>0:
#     putdraw = cv2.circle(rotated, (int(bott_right[r][0]),int(bott_right[r][1])), 2, (0,255,255), 2)


# cv2.imwrite("resmul_image.jpg", resmul_image)

# cv2.imwrite("putdraw.jpg", putdraw)


# putdraw_rotated = imutils.rotate_bound(putdraw, -90)
# putdraw_rotated2 = imutils.rotate_bound(rotated2, -90)
# cv2.imwrite("-90angle.jpg", putdraw_rotated)  #titik putih
# hasil akhir : posisi titik putih dan kuning , resmul dan end_resmul   pada hasil rotasi
# (resmul[0][1], ((putdraw_rotated.width-resmul[0][0]) )  = 229,20 =
# (resmul[1][1], ((putdraw_rotated.width-resmul[1][0]) )  = 218,31 =


# Hasil rotasi untuk menentukan potongan huruf, top left to bottom right coordinates
# Draw Bottom right square from r>0 list_split
# putdraw2 = cv2.circle(putdraw_rotated, (int(resmul[0][1]),int((putdraw_rotated.shape[0]-end_resmul[1][0]) )), 2, (255,0,255), 2)
# putdraw2 = cv2.circle(putdraw_rotated, (int(resmul[1][1]),int((putdraw_rotated.shape[0]-end_resmul[2][0]) )), 2, (255,0,255), 2)
# top left nya dari (resmul[0][1], int((putdraw_rotated.width-resmul[0][0]) )  = 229,20 =
# Bottom right dari (int(resmul[0][1]),int((putdraw_rotated.shape[0]-end_resmul[1][0])
# cv2.imwrite("putdraw2.jpg", putdraw2)

for r in range(len(rot_pos)):
    if r == 0:
        x2 = putdraw.shape[1]  # width
    else:  # r>0
        x2 = int(rot_pos[r - 1][0])

    y1 = int(rot_pos[r][1])
    y2 = int(rot_pos_end[r][1])

    x1 = int(rot_pos[r][0])

    print("x1,x2,y1,y2 ", x1, x2, y1, y2)  # 670 790 4 297
    # crop2= putdraw2[y1:y2, x1:x2]
    crop3 = rotated2[y1:y2, x1:x2]
    # putdraw_rotated3 = imutils.rotate_bound(crop3, 45)
    cv2.imwrite("crp-" + str(r) + ".jpg", crop3)


angle = -45
scale = 1.0
rotation_matrix2 = cv2.getRotationMatrix2D(center, angle, scale)
print("rotation_matrix2 ", rotation_matrix2)
print("center ", center)
resmul2 = np.matmul(res, rotation_matrix2)

print("resmul2= ", (resmul2))

print("resmul2 01 =", (int(resmul2[0][0]), int(resmul2[0][1])))


# mask = np.zeros(rotated.shape[:2], dtype="uint8")  #get partial
# cv2.rectangle(mask, (0, 0), (height, width), 255, -1) #top left   bottom right
# cv2.imshow("Rectangular Mask", mask)
# apply our mask
# cropped out
# masked = cv2.bitwise_and(rotated, rotated, mask=mask)
# cv2.imwrite("masked.jpg", masked)

back_rotated = imutils.rotate_bound(rotated, 45)
cv2.imwrite("back45angle.jpg", back_rotated)
