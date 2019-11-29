from keras import models, layers
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras


class LeNet:
    '''def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
        self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        self.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(layers.Flatten())
        self.add(layers.Dense(84, activation='tanh'))
        self.add(layers.Dense(nb_classes, activation='softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD')'''
    @staticmethod
    def build(width, height, channel, nb_class):
        input_shape = (width, height, channel)

        model = Sequential()
        model.add(Convolution2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(input_shape), padding="same"))
        model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(Convolution2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid', data_format='channels_first'))
        model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Convolution2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))
        model.add(Flatten())
        model.add(Dense(84, activation='tanh'))
        model.add(Dense(nb_class, activation='softmax'))

        return model