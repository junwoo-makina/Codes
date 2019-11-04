import cv2 as cv
import numpy as np

# default path setting
path = '/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/'
train_path = path + '1. Training'
class_path = path + '2. Class'
test_path  = path + '3. Test'

# Sobel algorithm

for j in range(0,10):
    image_source = cv.imread(train_path + '/Translate_image%d.bmp'%(j+1), cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(image_source, cv.COLOR_BGR2GRAY)
    '''
    img_sobel_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize = 3)
    img_sobel_x = cv.convertScaleAbs(img_sobel_x) # Vertical

    img_sobel_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize = 3)
    img_sobel_y = cv.convertScaleAbs(img_sobel_y) # Horizon

    sobel_image = cv.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)

    cv.imwrite('/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/sobel/%d_sobel.jpg'%j, sobel_image)
    '''

    edge_origin = cv.Canny(img_gray, 150, 200)
    cv.imwrite('/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/sobel/%d_canny.jpg'%j, edge_origin)
    # Morphology operation
    kernel = np.ones((11,11), np.uint8)
    result = cv.morphologyEx(edge_origin, cv.MORPH_CLOSE, kernel)
    cv.imwrite('/home/cheeze/PycharmProjects/KJW/research/defect_inspection/techwing/sobel/%d_close+sobel.jpg'%j, result)

