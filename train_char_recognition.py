
#! /usr/bin/env/python
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend

path_to_data = "augmented_plates/"

if __name__=="__main__":
    files = os.listdir("augmented_plates")
    
    #make datasets
    x_set = []
    y_set = []
    abc123 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for img_file in files:
        #make new blank one hot vector
        one_hot = np.zeros(36)
        for i, char in enumerate(abc123):
            if char == img_file[0]:
                one_hot[i] = 1
                break
        #append the current image and the one hot vector to sets
        x_set.append(cv2.imread(path_to_data + img_file))
        y_set.append(one_hot)
    x_set = np.array(x_set)
    y_set = np.array(y_set)

    VALIDATION_SPLIT = 0.2

    print("Total examples: {}\nTraining examples: {}\nTest examples: {}".
        format(x_set.shape[0],
                math.ceil(x_set.shape[0] * (1-VALIDATION_SPLIT)),
                math.floor(x_set.shape[0] * VALIDATION_SPLIT)))
    print("X shape: " + str(x_set.shape))
    print("Y shape: " + str(y_set.shape))

    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(135, 110, 3)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dropout(0.5))
    conv_model.add(layers.Dense(512, activation='relu'))
    conv_model.add(layers.Dense(36, activation='softmax'))
    conv_model.summary()
    LEARNING_RATE = 1e-4
    conv_model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                    metrics=['acc'])

    history_conv = conv_model.fit(x_set, y_set, 
                            validation_split=VALIDATION_SPLIT, 
                            epochs=20, 
                            batch_size=16)

    #save the model
    conv_model.save('NN_character_recognition')

    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()
            

    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    plt.show()
