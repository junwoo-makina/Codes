import cv2
import numpy as np

num = 200 #임계값 넘버
# 이미지 읽어오기
src = cv2.imread("/home/kanghee/Desktop/linedetecion/result/sobeltheth/hold%d.jpg"%num, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

for i in range(6):
    if i == 0:
        filsize = 3
    else:
        # 이미지 크기 변환
        filsize = i*2 + 1
    # 커널 생성
    kernel = np.ones((filsize, filsize), np.uint8)
    # para1 : 이미지, para2 : 커널, para3 : erode 반복 횟수
    erode = cv2.erode(gray, kernel, iterations=1)
    # para1 : 이미지, para2 : 커널, para3 : dilate 반복 횟수
    dilate = cv2.dilate(gray, kernel, iterations=1)
    # para1 : 이미지, para2 : 함수 이용, para3 : 커널
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # para1 : 이미지, para2 : 함수 이용, para3 : 커널
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("/home/kanghee/Desktop/linedetecion/result/mor/%d_%derd%d.jpg"%(i,num,filsize),erode)
    cv2.imwrite("/home/kanghee/Desktop/linedetecion/result/mor/%d_%ddil%d.jpg"%(i,num,filsize),dilate)
    cv2.imwrite("/home/kanghee/Desktop/linedetecion/result/mor/%d_%dopen%d.jpg"%(i,num,filsize),opening)
    cv2.imwrite("/home/kanghee/Desktop/linedetecion/result/mor/%d_%dclose%d.jpg"%(i,num,filsize),closing)
