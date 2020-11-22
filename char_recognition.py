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
from sklearn.metrics import confusion_matrix
import seaborn

if __name__=="__main__":
    path_to_data = "augmented_plates/"
    files = os.listdir("augmented_plates")
    
    #make datasets
    abc123 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    x_set = []
    y_set = []
    for img_file in files:
        x_set.append(cv2.imread(path_to_data + img_file))
        y_set.append(img_file[0])

    model = models.load_model('NN_character_recognition')
    
    predicted_chars = []
    for x_img in x_set:
        img_aug = np.expand_dims(x_img, axis=0)
        y_predicted = model.predict(img_aug)[0]
        max_val = np.amax(y_predicted)
        i = list(y_predicted).index(max_val)
        predicted_chars.append(abc123[i])
    cm = confusion_matrix(y_set, predicted_chars)
    seaborn.heatmap(cm, cmap="YlGnBu")
    plt.show()
