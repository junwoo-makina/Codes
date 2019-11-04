'''
Reference : https://ieeexplore.ieee.org/document/4147155
Implement : tensorflow 2.0, numpy, keras 2.0+,

Created by JunWoo Kim,
Date : 2019.4.3
'''
# Until line 138, implement by numpy array
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps
from PIL import Image
from glob import glob


'''Default path'''
train_path = '/home/cheeze/PycharmProjects/KJW/capstone_project/Image/'
image_files = os.listdir(train_path)
train_path_name = os.path.join(train_path, image_files[1])
total_file_paths = []

'''Get a whole paths : total_file_paths'''
for item in image_files:
    cur_file_path = os.path.join(train_path, item)
    total_file_paths = total_file_paths + [cur_file_path]

'''whole total_data_list'''
total_data_list = []
for item in total_file_paths:
    data_list = glob(os.path.join(train_path, item, '*'))
    total_data_list = total_data_list + data_list

'''Get Label'''
def get_label(path):
    return str(path.split('/')[-2])

label_list = get_label(total_data_list[1])



'''Get Random'''
rand_n = 100
print(total_data_list[rand_n], get_label(total_data_list[rand_n]))

'''Call Image&Label'''
path = total_data_list[rand_n]
image = np.array(Image.open(path))

def read_image(path):
   with open(path, 'rb') as f:
       with Image.open(f) as img:
           #img = ImageOps.grayscale(img)
           img = ImageOps.equalize(img)
           data = img.getdata()
           r=np.resize([(d[1]+d[2])/2 for d in data], 22500)
           g=np.resize([d[1]for d in data], 22500)
           b=np.resize([d[2] for d in data], 22500)
           rgb = [np.reshape(r, (150, 150)), np.reshape(g, (150, 150)), np.reshape(b, (150, 150))]

           return rgb

'''One hot encoding through the label name'''
class_name = get_label(path)
label_name_list = []
for path in total_data_list:
    label_name_list.append(get_label(path))

unique_label_names = np.unique(label_name_list)

def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label


train_label_list = []
for index in range(0, len(total_data_list)):
    train_label_list.append(onehot_encode_label(total_data_list[index]))

'''Hyper parameter'''
batch_size = 50
data_height = 150
data_width = 150
channel_n = 3

num_classes = len(image_files)

'''Setting tensor'''
batch_image = np.zeros((channel_n, data_height, data_width))
batch_label = np.zeros((num_classes))
#train_batch = tf.constant(batch_image)
#train_label = tf.constant(batch_label)


'''Create a convolutional base'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation ='relu'))
model.add(layers.Dense(1, activation ='softmax'))

print(model.summary())

model.compile(
    optimizer = 'sgd',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

'''Data collect from all folders'''
batch_per_epoch = len(total_data_list)//batch_size
for batch_n in range(0, batch_per_epoch):
    for n, path in enumerate(total_data_list[batch_n*batch_size:((batch_n+1)*batch_size)]):
        '''preprocessing with batch sizing, rescale image, extract label'''
        image = np.array(read_image(path))
        onehot_label = onehot_encode_label(path)
        plt.imshow(image)
        batch_image = np.hstack((batch_image, image))
        batch_label = np.hstack((batch_label, onehot_label))
        #train_tensor = tf.constant(image)
        #train_batch = tf.stack([train_batch, train_tensor])
        #batch_image[n, :, :, :] = batch_image + image
        #train_label = tf.stack([train_label, onehot_label])
        #batch_label[n, :] = onehot_label

'''train input vector/label'''
input = batch_image[1:]
label = bathch_label[1:]

model.fit(input, label, epochs = batch_per_epoch)

print(batch_image.shape, batch_label.shape)





