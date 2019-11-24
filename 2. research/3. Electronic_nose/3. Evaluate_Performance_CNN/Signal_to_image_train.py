import os
import numpy as np
import cv2
from keras.models import Sequential
from keras import layers
import keras
import glob

def parsing_data(path):
    f = open(path, 'r')
    data =[]
    label = []
    lines = f.readlines()
    for line in lines:
        x = line.split(' ')
        x = [float(x[i]) for i in range(len(x) -1)]
        data.append(x[:-1])
        label.append(x[-1])
    return data, label


def show_data(path, data, label):
    for i in range(len(data)):
        min = np.abs(np.min(data[i]))
        data[i] = data[i] + min
        max = np.max(data[i])
        data[i] = (data[i]/max)*255

        img = data[i].reshape((16, 2000))

        folder_num = str(np.where(label[i])[0][0])
        folder = path.replace('.dat', '')

        if not os.path.isdir(folder):
            os.mkdir(folder)
        if not os.path.isdir(os.path.join(folder, folder_num)):
            os.mkdir(os.path.join(folder, folder_num))
        cv2.imwrite(os.path.join(folder, folder_num, "data[%d].png"%i), img)


def z_normalize(x_train):#, x_test):
    for i in range(len(x_train)):
        min = np.abs(np.min(x_train[i]))
        x_train[i] = x_train[i] + min
        max = np.max(x_train[i])
        x_train[i] = (x_train[i] / max) * 255

    x_train = np.array(x_train)
    x_train = x_train.reshape((len(x_train), 160, 200))#2000, 16은 너무 꺠져버림

    train = []
    for i in range(len(x_train)):
        train.append(cv2.resize(x_train[i], (224, 224)).astype('int32'))

    return x_train


############################ Signal to Image ####################################
root_path = '/home/cheeze/PycharmProjects/KJW/1. Dataset/Dataset_recon_nose'
train_path_data = sorted(os.listdir(root_path))

for _, i in enumerate(train_path_data):
    iter_path = os.path.join(root_path, i)
    iters = sorted(os.listdir(iter_path))
    for _, j in enumerate(iters):
        set_path = os.path.join(iter_path, j)
        sets = sorted(os.listdir(set_path))
        for _, k in enumerate(sets):
            target_path = os.path.join(set_path, k)
            if target_path[-1] =='t':
                x_train, y_train = parsing_data(target_path)
                y_train = np.array(y_train)-1
                n_class = len(np.unique(y_train))
                x_train = z_normalize(x_train)
                y_train = keras.utils.to_categorical(y_train, n_class)
                show_data(target_path, x_train, y_train)

''' Image pre-processing Complete! '''
























