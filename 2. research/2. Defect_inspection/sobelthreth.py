import cv2
import numpy as np

#소벨 임계값 줫서 밝기 별로 뽑기

path = "/home/kanghee/Desktop/linedetecion/result/sobel3.jpg"

src = cv2.imread(path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
"""
for hold in range(255):
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            if int(gray[x][y]) > hold:
                gray[x][y] == 255
            else:
                gray[x][y] == 0
    cv2.imwrite("/home/kanghee/Desktop/linedetecion/result/sobeltheth/sobel%d.jpg" % hold, gray)
"""
new = np.ones((890, 2048))

for k in range(255):
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            if int(gray[x][y]) > int(k):
                new[x][y] = int(255)
            else:
                new[x][y] = int(0)

    cv2.imwrite("/home/kanghee/Desktop/linedetecion/result/sobeltheth/hold%d.jpg"%k, new)
print("heheheh")