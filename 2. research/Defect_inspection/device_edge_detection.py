import cv2 as cv
import numpy as np
import math
import time
import os

path= '/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/1. Training'
source_path = '/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/hough_transform_test'
for i in range(1,11):
    image_path = path + '/Translate_image%d.bmp'%i
    origin_img = cv.imread(image_path)
    cv.normalize(origin_img, origin_img, 0, 255, cv.NORM_MINMAX)
    gray = cv.cvtColor(origin_img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 70,90)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=70, maxLineGap=10)
    #lines = cv.HoughLines(edges, 1, np.pi/180, 500)

    # First method
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(origin_img, (x1,y1),(x2,y2),(0,255,0),2)

    ''' 
    # Second method
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(origin_img,(x1,y1),(x2,y2),(0,0,255),2)
    '''
    save_image_path = source_path + '/hough_transform_test_%d'%i
    cv.imwrite('transform_%d.jpg'%i, origin_img)
    cv.imwrite('canny_%d.jpg'%i, edges)
    cv.destroyAllWindows()
    #cv.imshow("canny", edges)
    #cv.imshow("result", origin_img)
    #k = cv.waitKey()
    #if k == 27:
    #    cv.destroyAllWindows()
    #elif k == ord('s'):
    #    cv.imwrite('transform_%d'%i, origin_img)
    #    cv.imwrite('canny_%d'%i, edges)
    #    cv.destroyAllWindows()
    #cv.imwrite('hough_transform_test_%d.jpg'%i, origin_img)


image_path2 = '/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/sample_canny.jpg'
test_img = cv.imread(image_path2)
cv.normalize(test_img, test_img, 0, 255, cv.NORM_MINMAX)
gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 70,90)
#lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=5, maxLineGap=10)
lines = cv.HoughLines(edges, 1, np.pi/180, 190)
test_img = test_img - 200
'''
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
'''

for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(test_img,(x1,y1),(x2,y2),(0,0,255),2)


cv.imshow("sobel", test_img)
cv.waitKey()
cv.destroyAllWindows()