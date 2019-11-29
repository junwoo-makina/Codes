import os
import numpy as np
import keras
from keras.models import Sequential
import resnet
from LeNet_implement import LeNet
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

def make_model():
    #base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', input_tensor=None,
     #                     pooling=None)
    #base_model = resnet.ResnetBuilder.build_resnet_18((3, 224, 224), 8)
    base_model = LeNet.build(224, 224, 3, 8)
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    #x = layers.GlobalAveragePooling2D()(x)  # averagePooling이 자동으로 되서 나오나본데?
    #x = layers.Dense(1024, activation='relu')(x)  # dense layer 3
    #preds = layers.Dense(8, activation='softmax')(x)  # final layer with softmax activation
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    # model.summary()

    return model

model = make_model()
test_datagen = ImageDataGenerator()
score = [0]
total = [[[0 for z in range(9)] for x in range(8)] for y in range(8)]

for i in range(8):
    for j in range(8):
        model.load_weights('/home/cheeze/PycharmProjects/KJW/2. Codes/2. research/3. Electronic_nose/3. Evaluate_Performance_CNN/weights/%d_%d_e-nose_2D.h5'%(i+1, j+1))
        sgd = keras.optimizers.SGD(lr=0.001, decay = 0.005, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print('iter%d set%d test'%(i+1,j+1))
        score=[0]
        for k in range(9): # for all loss per set
            test_generator = test_datagen.flow_from_directory(
                '/home/cheeze/PycharmProjects/KJW/1. Dataset/Dataset_electronic_nose/iter%d/set%d/set%d_test_data_loss_%d'%(i+1, j+1, j+1, (k+2)*5),
                shuffle=False,
                target_size=(224, 224),
                batch_size=20,
                color_mode='rgb',
                class_mode='categorical'
            )
            eval = model.evaluate_generator(test_generator, steps=1)
            pred = model.predict_generator(test_generator, steps=1)
            score += eval[1]
            print(eval[1])
            print(score)
            total[i][j][k] = eval[1]
            if(k==8):
                print('iter%d set%d average is %f'%(i+1, j+1, score[0] / 9))
                print('\n\n')

total = np.array(total)

for iters in range(8):
    for loss in range(9):  # (loss+2) *5
        loss_perform = 0
        for sets in range(8):
            loss_perform += total[iters,sets,loss]

        print("iter%d loss%d performance is : %f"%(iters+1, (loss+2)*5, loss_perform/8))


for loss in range(9):
    perform = 0
    for iters in range(8):
        perform = 0
        perform += np.sum(total[iters, :, loss])/8

    print("Loss %d data's performance : %f"%((loss+2)*5, perform))
