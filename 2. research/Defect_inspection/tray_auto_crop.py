import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time

image_path = '/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/1. Training'
start = time.time()
for i in range(0,10):
    print("time :", time.time() - start)
    image = cv.imread(image_path + '/Translate_image%d.bmp'%(i+1), cv.IMREAD_COLOR)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cnt = 0
    if not os.path.exists(image_path + '/_%dth_crop'%(i+1)):
        os.mkdir(image_path + '/_%dth_crop'%(i+1))

    #plt.imshow(image_gray, cmap='gray')
    #plt.show()

    for j in range(0, 12):
        # for horizon crop
        #plt.imshow(image_gray[10 + 73*(j) : 73*(j+1),:], cmap = 'gray')
        garo_image = image_gray[10 + 73 * j : 73 * (j + 1), :]
        for k in range(0, 20):
            #plt.imshow(garo_image[:,35 + 98 * k : 150 + 98 * k ], cmap= 'gray')
            crop_image = garo_image[:, 35 + 98 * k : 150 + 98 * k]
            cnt = cnt + 1
            cv.imwrite("/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/1. Training/_%dth_crop/"
                       "%d_crop.jpg"%(i+1, cnt), crop_image)

