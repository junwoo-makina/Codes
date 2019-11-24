import cv2 as cv
import numpy as np
import imageio
import scipy.ndimage
from skimage.measure import _structural_similarity

def dodge(front,back):
    result=front*255/(255-back+1)
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')

def gray(bgr):
    bgr[:,:,0] = bgr[:,:,0] * 0.1114
    bgr[:,:,1] = bgr[:,:,1] * 0.587
    bgr[:,:,2] = bgr[:,:,2] * 0.299
    return bgr

ssim = _structural_similarity.compare_ssim
path = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/'
category = ['cat', 'dog', 'pig']  # 1. cat, 2. dog, 3. pig

cat_path = path + category[0]
dog_path = path + category[1]
pig_path = path + category[2]
for i in range(1,12):
    pic_human = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_human/human_%04d.jpg'%(i)
    pic_cat = cat_path + '/human_cat_%04d.jpg'%(i)
    pic_dog = dog_path + '/human_dog_%04d.jpg'%(i)
    pic_pig = pig_path + '/human_pig_%04d.jpg'%(i)

    # Read Image
    img_human = cv.imread(pic_human)
    img_human = cv.resize(img_human, (256, 256))
    img_cat = cv.imread(pic_cat)
    img_dog = cv.imread(pic_dog)
    img_pig = cv.imread(pic_pig)

    # Canny algorithm
    #human_gray = cv.cvtColor(img_human, cv.COLOR_BGR2GRAY)
    #cat_gray = cv.cvtColor(img_cat, cv.COLOR_BGR2GRAY)
    #dog_gray = cv.cvtColor(img_dog, cv.COLOR_BGR2GRAY)
    #pig_gray = cv.cvtColor(img_pig, cv.COLOR_BGR2GRAY)

    human_gray = gray(img_human)
    cat_gray = gray(img_cat)
    dog_gray = gray(img_dog)
    pig_gray = gray(img_pig)

    human_gray2 = 255 - human_gray
    cat_gray2 = 255 - cat_gray
    dog_gray2 = 255 - dog_gray
    pig_gray2 = 255 - pig_gray

    #human_edges = cv.Canny(human_gray, 170, 190)
    #cat_edges = cv.Canny(cat_gray, 170, 190)
    #dog_edges = cv.Canny(dog_gray, 170, 190)
    #pig_edges = cv.Canny(pig_gray, 170, 190)

    # Gaussian Blurring
    human_gauss = cv.GaussianBlur(human_gray2, (5,5), 150)
    cat_gauss = cv.GaussianBlur(cat_gray2, (5,5), 150)
    dog_gauss = cv.GaussianBlur(dog_gray2, (5,5), 150)
    pig_gauss = cv.GaussianBlur(pig_gray2, (5,5), 150)

    # Dodge processing
    human_dodge = dodge(human_gauss, human_gray)
    cat_dodge = dodge(cat_gauss, cat_gray)
    dog_dodge = dodge(dog_gauss, dog_gray)
    pig_dodge = dodge(pig_gauss, pig_gray)

    cv.imwrite('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/cat/cat_%04d.jpg'%(i), cat_dodge)
    cv.imwrite('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/dog/dog_%04d.jpg'%(i), dog_dodge)
    cv.imwrite('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/pig/pig_%04d.jpg'%(i), pig_dodge)
    cv.imwrite('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/human/human_%04d.jpg'%(i), human_dodge)

    human = np.asarray(human_gauss)
    cat = np.asarray(cat_gauss)
    dog = np.asarray(dog_gauss)
    pig = np.asarray(pig_gauss)

    cat_err = np.sum((human_dodge.astype("float") - cat_dodge.astype("float"))**2)
    cat_err /= float(human_gauss.shape[0] * human_gauss.shape[1])
    cat_ssim = ssim(human_dodge, cat_dodge)

    dog_err = np.sum((human_dodge.astype("float") - dog_dodge.astype("float"))**2)
    dog_err /= float(human_gauss.shape[0] * human_gauss.shape[1])
    dog_ssim = ssim(human_dodge, dog_dodge)

    pig_err = np.sum((human_dodge.astype("float") - pig_dodge.astype("float"))**2)
    pig_err /= float(human_gauss.shape[0] * human_gauss.shape[1])
    pig_ssim = ssim(human_dodge, pig_dodge)


    print("The MSE of %04dth cat is : %f"%(i, cat_err))
    print("The MSE of %04dth dog is : %f"%(i, dog_err))
    print("The MSE of %04dth pig is : %f\n"%(i, pig_err))
    print("The SSIM of %04dth cat is : %f"%(i, cat_ssim))
    print("The SSIM of %04dth dog is : %f"%(i, dog_ssim))
    print("The SSIM of %04dth pig is : %f\n\n\n"%(i, pig_ssim))


