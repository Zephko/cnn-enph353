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
    train_set = ImageDataGenerator(
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
    rescale=1./255,
    preprocessing_function=None,
    data_format=None,
    # validation_split=0.0,
    # dtype="int",
    )

    val_set= ImageDataGenerator(
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
    rescale=1./255,
    preprocessing_function=None,
    data_format=None,
    # validation_split=0.0,
    # dtype="int",
    )


    train_gen = train_set.flow_from_directory(
    '../training_pictures/ped_train/',
    # target_size=(1280, 720),
    target_size=(150, 150),
    color_mode="rgb",
    # classes=['ped_N', 'ped_Y'],
    class_mode="binary",
    batch_size=16,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="jpg",
    follow_links=False,
    subset=None,
    interpolation="nearest",
    )
    
    val_gen = val_set.flow_from_directory(
    '../training_pictures/ped_val/',
    # target_size=(1280, 720)
    target_size=(150, 150),
    color_mode="rgb",
    # classes=['ped_N_val', 'ped_Y_val'],
    class_mode="binary",
    batch_size=16,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="jpg",
    follow_links=False,
    subset=None,
    interpolation="nearest",
    )
    return train_gen, val_gen

def setupNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
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
    model.add(layers.Dense(1, activation="softmax"))

    print(model.summary())

    LEARNING_RATE = 1e-4
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                    metrics=['acc'])
    return model

def fit_model(model, train_set, validation_set):
    print("lamb sauce")
    STEP_SIZE_TRAIN=train_set.n//train_set.batch_size
    STEP_SIZE_VALID=validation_set.n//validation_set.batch_size
    model_history = model.fit_generator(
        train_set,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=20,
        validation_data=validation_set,
        validation_steps=STEP_SIZE_VALID
    )
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()


if __name__=="__main__":
    train_set, val_set = preprocess_dataset()
    x,y = train_set.next()
    print(y)
    for i in range(0, 16):
        image = x[i]
        print(i)
        plt.imshow(image)
        plt.show()
    model = setupNN()
    fit_model(model, train_set, val_set)
