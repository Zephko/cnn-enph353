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
path = '../training_pictures/plate_sift/plate6.png'
BLUE_THRESH = 75E3 
abc123 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

class Plate():
    def __init__(self, img):
        self.model = models.load_model("../NN_character_recognition")
        self.img = img
        # self.slicePlate(img)
        self.findROIS(img)

    def findROIS(self, plate):

    ret, thresh = cv2.threshold(plate, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('thresh', thresh)

    im2, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        roi = img[y:y + h, x:x + w]

        area = w*h

        if 250 < area < 900:
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('rect', rect)

    def slicePlate(self, plate):

        lg_w = 108
        lg_h = 65
        lg_l_offset = 40
        lg_top= 40
        lg_bottom = 105

        sm_w = 45
        sm_h = 30
        sm_l_offset = 35
        sm_l2_offset = 173
        sm_top = 155
        sm_bottom = 185

        chars = []

        chars.append(plate[lg_top:lg_bottom, lg_l_offset: lg_l_offset + lg_w])
        chars.append(plate[lg_top:lg_bottom, lg_l_offset + lg_w: lg_l_offset + 2*lg_w])
        chars.append(plate[sm_top:sm_bottom, sm_l_offset:sm_l_offset + sm_w])
        chars.append(plate[sm_top:sm_bottom, sm_l_offset+sm_w:sm_l_offset+2*sm_w])
        chars.append(plate[sm_top:sm_bottom, sm_l2_offset:sm_l2_offset+sm_w])
        chars.append(plate[sm_top:sm_bottom, sm_l2_offset+sm_w:sm_l2_offset+2*sm_w])
        self.get_chars(chars)

    def get_chars(self, chars):
        #scale to correct size for NN?
        # for char in chars:
        #     PIL_image = Image.fromarray(np.uint8(char)).convert('RGB')
        #     PIL_image.resize((110, 135))
        # model = models.load_model("../NN_character_recognition")
        PIL_images = [Image.fromarray(np.uint8(char)).convert('RGB') for char in chars]
        PIL_images = [img.resize((110, 135)) for img in PIL_images]
        # imgs = [np.asarray(img) for img in PIL_images]
        plt.imshow(PIL_images[0])
        plt.show()
        # chars  = [cv2.resize(char, (135, 110)) for char in chars]

        for img in PIL_images:
            img_aug = np.expand_dims(img, axis=0)
            y_predicted = self.model.predict(img_aug)[0]
            max_val = np.amax(y_predicted)
            i = list(y_predicted).index(max_val)
            cv2.imwrite("../chars/{}_char.jpg".format(abc123[i]), np.asarray(img))


class Plate_matcher():

    def __init__(self, img_path, blue_threshold):
        self.img_path =img_path
        self.get_img_from_path()
        self.bridge = CvBridge()
        self.blue_threshold = blue_threshold
        self.num_plates = 0

    def get_img_from_path(self):
        self.img = cv2.imread(self.img_path)

    def read_camera(self, data):
        self.cam_img = self.bridge.imgmsg_to_cv2(data)
        blue = self.countBluePixels(self.cam_img)
        if blue > self.blue_threshold:
        # if self.countBluePixels(self.cam_img) > self.blue_threshold:
            self.blue = blue 
            self.get_plate()

    def get_plate(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(self.cam_img, cv2.COLOR_BGR2GRAY)
        height = np.shape(gray_img)[0]
        width = np.shape(gray_img)[1]
        gray_img = gray_img[:, :1780/3]
        gray_frame = gray_frame[:, :1780/3]
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(gray_img, None)
        kp2, desc2 = sift.detectAndCompute(gray_frame, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        #find good matches
        good_points = []
        min_matches = 8 
        for m, n in matches:
            if m.distance< 0.7 * n.distance:
                good_points.append(m)
  
        if len(good_points) > min_matches:
            
            query_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = gray_img.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            out_img = cv2.drawMatches(self.img, kp1, self.cam_img, kp2, good_points,self.cam_img)
            # plt.imshow(out_img)
            # plt.show()

            homography = cv2.polylines(self.cam_img, [np.int32(dst)], True, (255, 0, 0), 3)
            # plt.figure()
            # plt.imshow(homography)
            # plt.show()
            
            frame_pts = np.float32([[0,0],[0,width],[height,width], [height,0]])
            #check if transform is valid first, check that the 4 pts have right coords relative to each other
            corners = list(dst)
            print (corners)
            c_pts = [pt[0] for pt in corners]
            
            if (c_pts[0][1] < c_pts[1][1] and c_pts[0][0] < c_pts[3][0] and
                 c_pts[3][1] < c_pts[2][1] and c_pts[1][0] < c_pts[2][0] and 
                 c_pts[0][0] < c_pts[2][0] and c_pts[3][1] < c_pts[1][1]): 
                M = cv2.getPerspectiveTransform(dst, frame_pts)
                transformed = cv2.warpPerspective(homography, M, (height,width))
                cv2.imwrite("../sim_plates/{}_plate{}.jpg".format(self.num_plates, self.blue), transformed)
                plate = Plate(transformed)
                self.num_plates += 1
        
    def countBluePixels(self, img):
        img = img[:, :1780/3]
        mask1 = cv2.inRange(img, (0, 0, 90), (40, 40, 130))
        mask2 = cv2.inRange(img, (90, 90, 190), (105, 105, 210))
        mask = cv2.bitwise_or(mask1, mask2)
        output = cv2.bitwise_and(img, img, mask = mask)
        
        out = np.count_nonzero(output)
        print(out)
        return out
        # return np.count_nonzero(output)
    


if __name__ == "__main__":
    # for i in range(1, 7):
    #     img = cv2.imread("../training_pictures/plate_sift/plate{}.png".format(i))
    #     plate = Plate(img)
    plate_matcher = Plate_matcher(path, BLUE_THRESH)
    rospy.init_node('plate_matcher')
    rospy.Subscriber('R1/pi_camera/image_raw', ImageMsg, plate_matcher.read_camera)
    rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.spin()