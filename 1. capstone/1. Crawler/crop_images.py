import cv2
import numpy as np
import os
import glob

if __name__ == '__main__' :

    # Set a image file path
    base_path = '/home/cheeze/PycharmProjects/KJW/capstone_project/cabinet/downloads'
    cur_file_paths = glob.glob(os.path.join(base_path, ' Cow', '*'))
    cur_file_paths = sorted(cur_file_paths)
    for i, items in enumerate(cur_file_paths, 0):

        # Read image
        im = cv2.imread(items)

        # Select ROI
        fromCenter = False
        showCrosshair = False
        r = cv2.selectROI("Crop", im, fromCenter, showCrosshair)

        # Crop image
        imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # Display cropped image
        #cv2.imshow("Image", imCrop)
        cv2.imwrite('crop_cow_%d.jpg'%i, imCrop)


'''
How to crawl images from google?
- install chromedriver( version must be equal to your chrome version
- Following this github : https://github.com/hardikvasa/google-images-download#troubleshooting-errors-issues
- command
    pip install google_images_download
    googleimagesdownload --keywords "cow face" --limit 2000 --chromedriver /home/cheeze/PycharmProjects/KJW/capstone_project/cabinet/chromedriver (execute in terminal)

'''
