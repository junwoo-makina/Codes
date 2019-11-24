import numpy as np
import imageio
import scipy.ndimage
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import _structural_similarity
sns.set(style="darkgrid")

''' Define SSIM'''
ssim = _structural_similarity.compare_ssim

''' Path for humans'''
path = '/home/cheeze/PycharmProjects/KJW/capstone_project/' \
       'human2animal/transfer_network/image_transfer/test_human_crop/*.*'
path = sorted(glob.glob(path))
print(path)
'''
Two options,  1. test_human,       2. test_human_crop
'''

''' Path for animals'''
path2 = '/home/cheeze/PycharmProjects/KJW/capstone_project/' \
       'human2animal/transfer_network/image_transfer/test_transfer/'
category = ['cat/*.*', 'dog/*.*', 'pig/*.*']
cat_path = sorted(glob.glob(path2 + category[0]))
dog_path = sorted(glob.glob(path2 + category[1]))
pig_path = sorted(glob.glob(path2 + category[2]))


''' Convert RGB image to GRAY '''
#def gray(bgr):                         #
 #   bgr[:,:,0] = bgr[:,:,0] * 0.1114  #
  #  bgr[:,:,1] = bgr[:,:,1] * 0.587  #
   # bgr[:,:,2] = bgr[:,:,2] * 0.299 #
    #return bgr

def gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def dodge(front,back):
    result=front*255/(255-back+1)
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')



''' Calculate Image Histogram'''
for i in range(0,17):
    img_human= (cv.imread(path[i]))
    img_cat = (cv.imread(cat_path[i]))
    img_dog = (cv.imread(dog_path[i]))
    img_pig = (cv.imread(pig_path[i]))
    img_human = cv.resize(img_human, (256, 256))

    img_human_his = gray(img_human)
    img_cat_his = gray(img_cat)
    img_dog_his = gray(img_dog)
    img_pig_his = gray(img_pig)

    # Gaussian filtering
    human_gauss = scipy.ndimage.filters.gaussian_filter(255-img_human_his, 300)
    cat_gauss = scipy.ndimage.filters.gaussian_filter(255-img_cat_his, 300)
    dog_gauss = scipy.ndimage.filters.gaussian_filter(255-img_dog_his, 300)
    pig_gauss = scipy.ndimage.filters.gaussian_filter(255-img_pig_his, 300)

    # Calculate Dodge image
    human_dodge = cv.equalizeHist(dodge(human_gauss, img_human_his))  # Must be grayscale
    cat_dodge = cv.equalizeHist(dodge(cat_gauss, img_cat_his))
    dog_dodge = cv.equalizeHist(dodge(dog_gauss, img_dog_his))
    pig_dodge = cv.equalizeHist(dodge(pig_gauss, img_pig_his))


    # Calculate pixel-wise score
    cat_err = np.sum((human_dodge.astype("float") - cat_dodge.astype("float")) ** 2)
    cat_err /= float(human_dodge.shape[0] * human_dodge.shape[1])
    cat_ssim = ssim(human_dodge, cat_dodge)

    dog_err = np.sum((human_dodge.astype("float") - dog_dodge.astype("float")) ** 2)
    dog_err /= float(human_dodge.shape[0] * human_dodge.shape[1])
    dog_ssim = ssim(human_dodge, dog_dodge)

    pig_err = np.sum((human_dodge.astype("float") - pig_dodge.astype("float")) ** 2)
    pig_err /= float(human_dodge.shape[0] * human_dodge.shape[1])
    pig_ssim = ssim(human_dodge, pig_dodge)

    # Get error
    min_error = min(cat_err, dog_err, pig_err)
    label = (lambda n,m,k : 'Cat' if(n==min_error) else 'Dog' if(m==min_error) else 'Pig')(cat_err, dog_err, pig_err)

    # Calculate & Compare histogram
    hist_human = cv.calcHist([img_human], [1], None, [256], [0,256])
    hist_cat = cv.calcHist(img_cat, [1], None, [256], [0,256]) * 100
    hist_dog = cv.calcHist(img_dog, [1], None, [256], [0,256]) * 100
    hist_pig = cv.calcHist(img_pig, [1], None, [256], [9,256]) * 100
    #cv.equalizeHist(hist_human)
    #cv.equalizeHist(hist_cat)
    #cv.equalizeHist(hist_dog)
    #cv.equalizeHist(hist_pig)



    #plt.subplot(221), plt.plot(hist_human, color='r'), plt.title(label), plt.xlim([0,255]), plt.ylim([0,3000])
    #plt.subplot(222), plt.plot(hist_cat, color = 'g'), plt.title('cat'), plt.xlim([0,255]), plt.ylim([0,3000])
    #plt.subplot(223), plt.plot(hist_dog, color = 'b'), plt.title('dog'), plt.xlim([0,255]), plt.ylim([0,3000])
    #plt.subplot(224), plt.plot(hist_pig, color = 'k'), plt.title('pig'), plt.xlim([0,255]), plt.ylim([0,3000])

    f, axes = plt.subplots(2,2, figsize=(8,6), sharex=True)
    sns.distplot(hist_human, color="y", axlabel=label, ax=axes[0,0]), plt.xlim([0,2000])
    sns.distplot(hist_cat, color="r", axlabel="Cat_face", ax=axes[0,1]),  plt.xlim([0,2000])
    sns.distplot(hist_dog, color="b", axlabel="Dog_face", ax=axes[1,0]),  plt.xlim([0,2000])
    sns.distplot(hist_pig, color="g", axlabel="Pig_face", ax=axes[1,1]),  plt.xlim([0,2000])
    #plt.show()
    #plt.xlim([0,400])
    #plt.show()
    plt.savefig('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/plot/plot%04d.png'%(i))
    plt.close()
    '''
    plt.subplot(221), plt.imshow(img1, 'gray'), plt.title('Red Line')
    plt.subplot(222), plt.imshow(img2, 'gray'), plt.title('Green Line')
    plt.subplot(223), plt.plot(hist1, color='r'), plt.plot(hist2, color='g')
    plt.xlim([0, 256])
    plt.show()
    '''

    '''
    human = cv.cvtColor(img_human, cv.COLOR_BGR2RGB)
    cat = cv.cvtColor(img_cat, cv.COLOR_BGR2RGB)
    dog = cv.cvtColor(img_dog, cv.COLOR_BGR2RGB)
    pig = cv.cvtColor(img_pig, cv.COLOR_BGR2RGB)

    # human-cat blending
    hucat_0 = cv.addWeighted(human, float(100-0)/100, cat, float(0)/100, 0)
    hucat_1 = cv.addWeighted(human, float(100-33)/100, cat, float(33)/100, 0)
    hucat_2 = cv.addWeighted(human, float(100-66)/100, cat, float(66)/100, 0)
    hucat_3 = cv.addWeighted(human, float(100-100)/100, cat, float(100)/100, 0)

    plt.imsave('hucat_0 path') # You can indicate the path.
    plt.imsave('hucat_1 path')  # You can indicate the path.
    plt.imsave('hucat_2 path')  # You can indicate the path.
    plt.imsave('hucat_3 path')  # You can indicate the path.

    # human-dog blending
    hudog_0 = cv.addWeighted(human, float(100 - 0) / 100, dog, float(0) / 100, 0)
    hudog_1 = cv.addWeighted(human, float(100 - 33) / 100, dog, float(33) / 100, 0)
    hudog_2 = cv.addWeighted(human, float(100 - 66) / 100, dog, float(66) / 100, 0)
    hudog_3 = cv.addWeighted(human, float(100 - 100) / 100, dog, float(100) / 100, 0)

    plt.imsave('hudog_0 path')  # You can indicate the path.
    plt.imsave('hudog_1 path')  # You can indicate the path.
    plt.imsave('hudog_2 path')  # You can indicate the path.
    plt.imsave('hudog_3 path')  # You can indicate the path.

    # human-pig blending
    hupig_0 = cv.addWeighted(human, float(100 - 0) / 100, pig, float(0) / 100, 0)
    hupig_1 = cv.addWeighted(human, float(100 - 33) / 100, pig, float(33) / 100, 0)
    hupig_2 = cv.addWeighted(human, float(100 - 66) / 100, pig, float(66) / 100, 0)
    hupig_3 = cv.addWeighted(human, float(100 - 100) / 100, pig, float(100) / 100, 0)

    plt.imsave('hupig_0 path')  # You can indicate the path.
    plt.imsave('hupig_1 path')  # You can indicate the path.
    plt.imsave('hupig_2 path')  # You can indicate the path.
    plt.imsave('hupig_3 path')  # You can indicate the path.
    '''