#!/usr/bin/python

import os, sys

import math
import numpy as np
import re
import cv2

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image 

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator

raw_image_dir = "/home/fizzer/ros_ws/src/2020T1_competition/enph353/cnn-enph353/raw_plates/pictures"
augmented_plates_dir = "/home/fizzer/ros_ws/src/2020T1_competition/enph353/cnn-enph353/augmented_plates"
count = np.zeros(36)


def load_images_from_folder(folder):
    images = []
    image_names = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            image_names.append(filename.split('_')[1].split('.')[0])
    return images, image_names

def update_count(char):
    global count
    if char.isnumeric():
        count[26 + int(char)] += 1
        return count[26 + int(char)]
    else:
        count[ord(char) - 65] += 1
        return count[ord(char) - 65]

if __name__ == '__main__':
    global count
    raw_images, image_names = load_images_from_folder(raw_image_dir)
    augmenter = ImageDataGenerator(brightness_range=[0.4,1.2],width_shift_range=[-8,8],height_shift_range=0.2,rotation_range=5,zoom_range=[1,1.5])

    # print(len(raw_images))
    # print(image_names)

    for index in np.arange(0, len(raw_images)):
    	img_arr = []
    	img_arr.append(raw_images[index][90:225,40:150])
    	img_arr.append(raw_images[index][90:225,140:250])
    	img_arr.append(raw_images[index][90:225,340:450])
    	img_arr.append(raw_images[index][90:225,440:550])

    	for j in range(0,4):
    	    samples = expand_dims(img_arr[j], 0)
    	    it = augmenter.flow(samples, batch_size=1)
    	    batch = it.next()
    	    image = batch[0].astype('uint8')

    	    new_count = update_count(unicode(image_names[index][j],'utf-8'))
            path = augmented_plates_dir + "/" + image_names[index][j] + "_" + str(new_count).split(".")[0] + ".png"
            cv2.imwrite(path, image)

            new_count = update_count(unicode(image_names[index][j],'utf-8'))
            path = augmented_plates_dir + "/" + image_names[index][j] + "_" + str(new_count).split(".")[0] + ".png"
            cv2.imwrite(path, img_arr[j])

    print(count)


        