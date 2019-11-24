import os
import numpy as np
import cv2
from keras.models import Sequential
from keras import layers
from keras_radam import RAdam
import keras

path = r'D:\e-nose\iter1\set1\set1_tr_data.dat'
target = r'D:\e-nose\iter1\set1\set1_test_data_loss_5.dat'

def make_model_GRU(input_dim, output_dim, length):#sigmoid 값을 예측할순 없고, categorical이랑 sigmoid를 어떻게 해결할 수 있도록 하는지..
    model = Sequential()
    #model.add(layers.Embedding(input_dim, output_dim, input_length = length))
    model.add(layers.LSTM(output_dim, input_shape = (None, 16), dropout = 0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.LSTM(output_dim*2, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
    model.add(layers.LSTM(output_dim*4, dropout=0.2, recurrent_dropout=0.2))
    #model.add(layers.LSTM(output_dim, dropout=0.2, recurrent_dropout=0.2))

    #model.add(layers.Flatten())

    model.add(layers.Dense(8, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = RAdam(), metrics=["accuracy"])
    model.summary()
    return model

def make_model_1D(input_dim, output_dim, length):
    model = Sequential()

    model.add(layers.Conv1D(output_dim, 5, activation = 'relu', input_shape = (None, input_dim)))#stride 정해줘야함
    model.add(layers.Conv1D(output_dim, 5, activation='relu', input_shape=(None, input_dim)))
    model.add(layers.Conv1D(output_dim, 5, activation='relu', input_shape=(None, input_dim)))
    model.add(layers.Conv1D(output_dim, 5, activation='relu', input_shape=(None, input_dim)))
    model.add(layers.Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RAdam)
    model.summary()
    return model()

def parsing_data(path):
    f = open(path, 'r')
    data = []
    label = []
    lines = f.readlines()
    for line in lines:
        x = line.split(' ')
        x = [float(x[i]) for i in range(len(x) - 1)]
        data.append(x[:-1])
        label.append(int(x[-1]))
    return data, label

def show_data(data):

    for i in range(len(data)):
        min = np.abs(np.min(data[i]))
        data[i] = data[i]+min
        max = np.max(data[i])
        data[i] = (data[i] / max) * 255

        img = data[i].reshape((16, 2000))

        cv2.imwrite("data[%d].jpg" %i, img)
        #cv2.waitKey(1000)

def z_normalize(x_train, x_test):
    x_train = np.array(x_train)
    x_teste = np.array(x_test)

    mean = np.mean(x_train)
    std = np.std(x_test)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    x_train = x_train.reshape((len(x_train), 2000, 16))
    x_test = x_test.reshape((len(x_test), 2000, 16))

    return x_train, x_test

x_train, y_train = parsing_data(path)
x_test, y_test = parsing_data(target)

y_train = np.array(y_train) -1
y_test = np.array(y_test) -1

n_class = len(np.unique(y_train))

y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)

x_train, x_test =  z_normalize(x_train, x_test)
#show_data(data)

model = make_model_GRU(len(x_train[0]), 8, len(x_train[0][0]))
model.fit(x=x_train, y=y_train, epochs = 100, batch_size = 32, verbose = 1)
