#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import sys
from keras import models
from keras import backend
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image

# path = '../training_pictures/plate_sift/plate_blue.png'
# path = '../training_pictures/plate_sift/plate6.png'
BLUE_THRESH = 75E3 
abc123 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
class Plate():

    def __init__(self, img):
        self.img = img
        # self.slicePlate(img)
        self.findROIS(img)

    def findROIS(self, plate):

        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_OTSU)
        plt.imshow(thresh)
        plt.show()

        im2, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        print(hier)
        print(len(ctrs))

        pad = 5 
        area_low_bound = 500 
        area_high_bound = 50000
        common_parent = max([ctr[3] for ctr in hier[0]])
        rois = []
        for i, ctr in enumerate(ctrs):
            x, y, w, h = cv2.boundingRect(ctr)

            area = w*h
            
            #requires parent to be the largest frame and have no children and be large enough to be a char
            if hier[0][i][2] == -1 and hier[0][i][3] == common_parent and area_low_bound< area < area_high_bound:
                # rect = cv2.rectangle(img, (x-pad, y-pad), (x + w+pad, y + h+pad), (0, 255, 0), 2)
                roi = img[y-pad:y + h+ pad, x-pad:x + w + pad]
                rois.append(roi)
                plt.imshow(roi)
                plt.show()
        self.get_chars(rois)

    def get_chars(self, chars):
        #scale to correct size for NN?
        # for char in chars:/)
        #     PIL_image = Image.fromarray(np.uint8(char)).convert('RGB')
        #     PIL_image.resize((110, 135))
        PIL_images = [Image.fromarray(np.uint8(char)).convert('RGB') for char in chars]
        PIL_images = [img.resize((110, 135)) for img in PIL_images]
        plt.imshow(PIL_images[0])
        plt.show()
        # chars  = [cv2.resize(char, (135, 110)) for char in chars]

        for img in PIL_images:
            img_aug = np.expand_dims(img, axis=0)
            y_predicted = model.predict(img_aug)[0]
            max_val = np.amax(y_predicted)
            i = list(y_predicted).index(max_val)
            cv2.imwrite("chars/{}_char.jpg".format(abc123[i]), np.asarray(img))

if __name__ == "__main__":
    model = models.load_model("NN_character_recognition")
    img = cv2.imread("training_pictures/plate_sift/plate1.png")
    plate = Plate(img)