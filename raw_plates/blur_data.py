#!/usr/bin/python

import os, sys

import math
import numpy as np
import re
import cv2

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image 

import skimage
from skimage.viewer import ImageViewer
import sys
from skimage import img_as_ubyte
# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator

raw_image_dir = "/home/fizzer/ros_ws/src/2020T1_competition/enph353/cnn-enph353/augmented_plates"
augmented_plates_dir = "/home/fizzer/ros_ws/src/2020T1_competition/enph353/cnn-enph353/blurred_augmented_plates"
count = np.zeros(36)


def load_images_from_folder(folder):
    images = []
    image_names = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            image_names.append(filename)
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
    raw_images, image_names = load_images_from_folder(raw_image_dir)

    print(len(raw_images))
    print(image_names[0:10])

    for index in range(0, len(raw_images)):

    	blurred = skimage.filters.gaussian(raw_images[index], sigma=(7,7), truncate=3.5, multichannel=True)

    	cv_image = img_as_ubyte(blurred)

        path = augmented_plates_dir + "/" + image_names[index] +"_blurred.png"
        cv2.imwrite(path, cv_image)



        