#! /usr/bin/env/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend
# from tensorflow import keras
# from keras.preprocessing.image import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator

def preprocess_dataset():
    # return keras.preprocessing.image_dataset_from_directory(
    #     '../training_pictures/ped_sets',
    #     labels="inferred",
    #     label_mode="binary",
    #     class_names=["ped_N", "ped_Y"],
    #     color_mode="rgb",
    #     batch_size= 50,
    #     image_size=(1280, 720),
    #     shuffle=False,
    #     validation_split= 0.2,
    #     subset="training",
    # )
    idg = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    # dtype="int",
    )


    return idg.flow_from_directory(
    '../training_pictures/ped_sets/',
    # target_size=(1280, 720),
    target_size=(720, 1280),
    color_mode="rgb",
    classes=['ped_N', 'ped_Y'],
    class_mode="binary",
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="jpg",
    follow_links=False,
    subset=None,
    interpolation="nearest",
)

def setupNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation="softmax"))

    LEARNING_RATE = 1e-4
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                    metrics=['acc'])
    print(model.summary())


if __name__=="__main__":
    # iterator = preprocess_dataset()
    # x,y = iterator.next()
    # for i in range(0, 1):
    #     image = x[1]
    #     plt.imshow(image)
    #     plt.show()
    setupNN()