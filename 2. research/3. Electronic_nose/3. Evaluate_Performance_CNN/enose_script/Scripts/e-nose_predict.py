import os
import os
import numpy as np
from keras.models import Sequential
from keras import layers
#from keras_radam import RAdam
import keras
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

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

model = make_model()

test_datagen = ImageDataGenerator(
)

score = [0]
#score = [[0] for i in range(9)]
#score = [[0] for i in range(10)]

for i in range(8):
    for j in range(8):

        model.load_weights(r'C:\Users\User\PycharmProjects\e-nose\x-ray\Scripts\weight\%d_%d_e-nose_2D.h5' % (i+1, j+1))
        sgd = keras.optimizers.SGD(lr=0.001, decay=0.005, momentum=0.9)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #for k in range(9):
            #test_generator = test_datagen.flow_from_directory(
            #r'D:\e-nose\autoencoder\iter%d\set%d\set%d_test_data_loss_%d_recon_DAE'%(i+1, j+1, j+1, (k+2)*5),
        #for k in range(10):
        #test_generator = test_datagen.flow_from_directory(
        #r'D:\e-nose\original\iter%d\set%d\set%d_test_data_loss_%d'%(i+1, j+1, j+1, (k+1)*5),
        test_generator = test_datagen.flow_from_directory(
        r'D:\e-nose\lossless_data\iter%d\set%d\set%d_test_data'%(i+1, j+1, j+1),
        shuffle=False,
        target_size=(224, 224),
        batch_size=20,
        color_mode='rgb',
        class_mode='categorical')

        eval = model.evaluate_generator(test_generator, steps=1)
        pred = model.predict_generator(test_generator, steps=1)
        score += eval[1]

        print(score)