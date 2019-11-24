import os
import numpy as np
import cv2
from keras.models import Sequential
from keras import layers
#from keras_radam import RAdam
import keras
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet18 import ResNet18
from keras.preprocessing.image import ImageDataGenerator

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

def normalize(x_train, x_test):

    for i in range(len(x_train)):
        min = np.abs(np.min(x_train[i]))
        x_train[i] = x_train[i] + min
        max = np.max(x_train[i])
        x_train[i] = (x_train[i] / max) * 255

    for i in range(len(x_test)):
        min = np.abs(np.min(x_test[i]))
        x_test[i] = x_test[i] + min
        max = np.max(x_test[i])
        x_test[i] = (x_test[i] / max) * 255

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    #x_train = x_train.reshape((len(x_train), 2000, 16))
    #x_test = x_test.reshape((len(x_test), 2000, 16))

    x_train = x_train.reshape((len(x_train), 160, 200))#2000, 16은 너무 꺠져버림
    x_test = x_test.reshape((len(x_test), 160, 200))

    train = []
    for i in range(len(x_train)):
        train.append(cv2.resize(x_train[i], (224, 224)).astype('int32'))

    test = []
    for i in range(len(x_test)):
        test.append(cv2.resize(x_test[i], (224, 224)).astype('int32'))

    return train, test

def make_model():
    #base_model = NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet', input_tensor=None,pooling=None)
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', input_tensor=None,
                              pooling=None)

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)  # averagePooling이 자동으로 되서 나오나본데?
    x = layers.Dense(1024, activation='relu')(x)  # dense layer 3
    preds = layers.Dense(8, activation='softmax')(x)  # final layer with softmax activation
    model = keras.models.Model(inputs=base_model.input, outputs=preds)
    #model.summary()

    return model

train_datagen = ImageDataGenerator(
)

test_datagen = ImageDataGenerator(
)

root = '/home/cheeze/PycharmProjects/KJW/1. Dataset/e-nose/e-nose/original_img'
target = r'D:\e-nose\autoencoder'

test_str = ['_test_data_loss_%d'%(i*5) for i in range(1,11)]
dae_str = ['_test_data_loss_%d_recon_DAE'%(i*5) for i in range(2, 11)]

root_folder = os.listdir(root)

s_score = [[0] for i in range(9)]

for _, i in enumerate(root_folder):# i == iter
    first_path = os.path.join(root, i)

    first_folder = os.listdir(first_path)

    for __, j in enumerate(first_folder): # j == set
        second_path = os.path.join(first_path, j)

        second_folder = os.listdir(second_path)

        model = make_model()


        sgd = keras.optimizers.SGD(lr=0.001, decay=0.005, momentum=0.9)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        train_generator = train_datagen.flow_from_directory(
            os.path.join(root, i, j, j + '_tr_data'),
            shuffle=True,
            target_size=(224, 224),
            batch_size=16,
            color_mode='rgb',
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch= 140 // 32,
            validation_data=None,
            validation_steps=None,
            epochs=50, verbose=1)

        model.save_weights(os.path.join(r"./%d_%d_e-nose_2D.h5" % (_+1, __+1)))

        for k in range(9):
            test_generator = test_datagen.flow_from_directory(
                os.path.join(target, i, j, j + dae_str[k]),
                shuffle=True,
                target_size=(224, 224),
                batch_size=16,
                color_mode='rgb',
                class_mode='categorical')

            score = model.evaluate_generator(test_generator, steps=1)
            s_score[k] += score[1]

            pred = model.predict_generator(test_generator, steps=1)
            print(s_score)

        keras.backend.clear_session()
        """
        for k in range(10):
            train_generator = train_datagen.flow_from_directory(
                os.path.join(root, i, j, j+'_tr_data'),
                shuffle=True,
                target_size=(224, 224),
                batch_size=32,
                color_mode='rgb',
                class_mode='categorical')

            test_generator = test_datagen.flow_from_directory(
                os.path.join(target, i, j, j+dae_str[k]),
                shuffle=True,
                target_size=(224, 224),
                batch_size=20,
                color_mode='rgb',
                class_mode='categorical')
        
            model.compile(loss = 'categorical_crossentropy', optimizer = RAdam())

            model.fit_generator(
                train_generator,
                steps_per_epoch= 140 // 32,
                epochs=1, verbose=1)

            model.save(os.path.join(r"./%d_%d_e-nose_2D.h5"%(_, __)))

            model.predict_generator(test_generator, steps = 1)
        """