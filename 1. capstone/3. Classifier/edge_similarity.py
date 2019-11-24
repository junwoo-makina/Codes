import numpy as np
import imageio
import scipy.ndimage
from skimage.measure import _structural_similarity
from skimage.transform import resize
import matplotlib.pyplot as plt


def dodge(front,back):
    result=front*255/(255-back+1)
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')

def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

ssim = _structural_similarity.compare_ssim
path = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/'
category = ['cat', 'dog', 'pig']

cat_path = path + category[0]
dog_path = path + category[1]
pig_path = path + category[2]

for i in range(1,18):
    pic_human = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/human/human_%04d.jpg'%(i)
    pic_cat = cat_path + '/human_cat_%04d.jpg'%(i)
    pic_dog = dog_path + '/human_dog_%04d.jpg'%(i)
    pic_pig = pig_path + '/human_pig_%04d.jpg'%(i)

    # Read Image
    img_human = imageio.imread(pic_human)
    img_human = np.asarray(img_human)
    img_human = resize(img_human, (256,256,3)) *256

    img_cat = imageio.imread(pic_cat)
    img_dog = imageio.imread(pic_dog)
    img_pig = imageio.imread(pic_pig)

    human_gray = grayscale(img_human)
    cat_gray = grayscale(img_cat)
    dog_gray = grayscale(img_dog)
    pig_gray = grayscale(img_pig)

    human_gauss = scipy.ndimage.filters.gaussian_filter(255-human_gray, 300)
    cat_gauss = scipy.ndimage.filters.gaussian_filter(255-cat_gray, 300)
    dog_gauss = scipy.ndimage.filters.gaussian_filter(255-dog_gray, 300)
    pig_gauss = scipy.ndimage.filters.gaussian_filter(255-pig_gray, 300)

    human_dodge = dodge(human_gauss, human_gray)
    cat_dodge = dodge(cat_gauss, cat_gray)
    dog_dodge = dodge(dog_gauss, dog_gray)
    pig_dodge = dodge(pig_gauss, pig_gray)

    plt.imsave('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/cat/cat_%04d.jpg'%(i), cat_dodge, cmap ='gray', vmin=0, vmax=255)
    plt.imsave('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/dog/dog_%04d.jpg'%(i), dog_dodge, cmap ='gray', vmin=0, vmax=255)
    plt.imsave('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/pig/pig_%04d.jpg'%(i), pig_dodge, cmap ='gray', vmin=0, vmax=255)
    plt.imsave('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/edge_result/human/human_%04d.jpg'%(i), human_dodge, cmap ='gray', vmin=0, vmax=255)

    cat_err = np.sum((human_dodge.astype("float")-cat_dodge.astype("float"))**2)
    cat_err /= float(human_gray.shape[0] * human_gray.shape[1])
    cat_ssim = ssim(human_dodge, cat_dodge)

    dog_err = np.sum((human_dodge.astype("float")-dog_dodge.astype("float"))**2)
    dog_err /= float(human_gray.shape[0] * human_gray.shape[1])
    dog_ssim = ssim(human_dodge, dog_dodge)

    pig_err = np.sum((human_dodge.astype("float")-pig_dodge.astype("float"))**2)
    pig_err /= float(human_gray.shape[0] * human_gray.shape[1])
    pig_ssim = ssim(human_dodge, pig_dodge)

    print("The MSE of %04dth cat is : %f" % (i, cat_err))
    print("The MSE of %04dth dog is : %f" % (i, dog_err))
    print("The MSE of %04dth pig is : %f\n" % (i, pig_err))
    print("The SSIM of %04dth cat is : %f" % (i, cat_ssim))
    print("The SSIM of %04dth dog is : %f" % (i, dog_ssim))
    print("The SSIM of %04dth pig is : %f\n\n\n" % (i, pig_ssim))
