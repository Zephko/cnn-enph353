#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import sys
import time
from keras import models
from keras import backend
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
from std_msgs.msg import String, Bool, Int32

# path = '../training_pictures/plate_sift/plate_blue.png'i
path = '../training_pictures/plate_sift/'
paths = [path + 'plate2.png',
        path + 'plate3.png',
        path + 'plate4_test.png',
        path + 'plate5.png',
        path + 'plate6_test.png',
        path + 'plate1_test.png',
        path + 'plate7.png',
        path + 'plate8.png',
        path + 'plate8.png']
stall_order = [2, 3, 4, 5, 6, 1, 7, 8]
BLUE_THRESH = 95E3 #75E3
abc123 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TIME_TO_PASS_CAR = 5
do_sift = True
# nn_model = models.load_model("../NN_character_recognition_blurred")

class Plate():
    # def __init__(self, img, car_num, model):
    def __init__(self, img, car_num):
        self.model = models.load_model("../NN_character_recognition_blurred")
        # self.model = model
        # self.model._make_predict_function()
        self.img = img
        self.car_num = car_num 
        self.findROIs(img)

    def findROIs(self, plate):

        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_OTSU)
        # plt.imshow(thresh)
        im2, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        pad = 5 
        area_low_bound = 500 
        area_high_bound = 50000
        common_parent = max([ctr[3] for ctr in hier[0]])
        rois = []
        print("# of Contours:{}".format(len(ctrs)))
        if len(ctrs) > 5:
            for i, ctr in enumerate(ctrs):
                x, y, w, h = cv2.boundingRect(ctr)

                area = w*h
                
                #requires parent to be the largest frame and have no children and be large enough to be a char
                if hier[0][i][2] == -1 and hier[0][i][3] == common_parent and area_low_bound< area < area_high_bound:
                    # rect = cv2.rectangle(img, (x-pad, y-pad), (x + w+pad, y + h+pad), (0, 255, 0), 2)
                    roi = plate[y-pad:y + h+ pad, x-pad:x + w + pad]
                    rois.append((roi, cv2.boundingRect(ctr)))
                    # rois.append((roi, ctr))
                    # plt.imshow(roi)
                    # plt.show()
            # origins = [roi[1] for roi in rois]
            if len(rois) > 5:
                print("enough regions of interest")
                vert_sorted_rois = sorted(rois, key=lambda x: x[1][1])
                stall = sorted(vert_sorted_rois[:2], key=lambda x:x[1][0])
                # for char in stall:
                #     plt.imshow(char[0])
                #     plt.show()
                # print(stall)
                stall_num = stall[1][0]
                hor_sorted_rois = sorted(vert_sorted_rois[2:], key=lambda x: x[1][0])

                if hor_sorted_rois > 3:
                    plate_nums = [roi[0] for roi in hor_sorted_rois]
                    # for roi in hor_sorted_rois:
                    #     plt.imshow(roi[0])
                    #     plt.show()
                    # for roi in sorted_rois:
                    #     plt.imshow(roi[0])
                    #     plt.show()
                    print("# Recognized ROIs:{}".format(len(vert_sorted_rois)))
                    # self.get_chars([roi[0] for roi in sorted_rois])
                    self.get_chars(stall_num, plate_nums)

    def get_chars(self, stall, plate):
        print("making prediction")
        #scale to correct size for NN
        plate = [cv2.resize(char, (110, 135)) for char in plate]
        stall  = cv2.resize(stall, (110, 135)) 
        predicted_plate_chars = []
        # if len(chars) == 6:
        #predict plate nums
        for i, img in enumerate(plate):
            img_aug = np.expand_dims(img, axis=0)
            y_predicted = self.model.predict(img_aug)[0]
            if i < 2:
                y_predicted_letters = y_predicted[:26]
                max_val = np.amax(y_predicted_letters)
            else:
                y_predicted_nums = y_predicted[26:]
                max_val = np.amax(y_predicted_nums)
            # max_val = np.amax(y_predicted)
            i = list(y_predicted).index(max_val)
            predicted_plate_chars.append(abc123[i])
            # cv2.imwrite("../chars/{}_char_{}.jpg".format(abc123[i], self.car_num), np.asarray(img)
        self.plate_nums = predicted_plate_chars
        
        #predict stall num
        img_aug = np.expand_dims(stall, axis=0)
        y_predicted = self.model.predict(img_aug)[0]
        max_val = np.amax(y_predicted)
        i = list(y_predicted).index(max_val)
        cv2.imwrite("../chars/{}_char.jpg".format(abc123[i]), np.asarray(stall))
        self.stall_num = abc123[i]

        self.publish_plate()

        # cv2.imwrite("../chars/{}_char.jpg".format(abc123[i]), np.asarray(img))
        # print(predictions)
        # print("wrote {} chars to file".format(len(chars)))
    # def get_chars(self, chars):
    #     #scale to correct size for NN
    #     chars  = [cv2.resize(char, (110, 135)) for char in chars]

    #     predictions = []
    #     if len(chars) == 6:
    #         for img in chars:
    #             img_aug = np.expand_dims(img, axis=0)
    #             y_predicted = self.model.predict(img_aug)[0]
    #             max_val = np.amax(y_predicted)
    #             i = list(y_predicted).index(max_val)
    #             predictions.append(abc123[i])
    #             cv2.imwrite("../chars/{}_char.jpg".format(abc123[i]), np.asarray(img))
    #         print(predictions)
    #         print("wrote {} chars to file".format(len(chars)))
    
    def publish_plate(self):
        global do_sift
        print(self.stall_num)
        print(str(self.plate_nums))
        # plate_publisher.publish("funMode,passwd,{},{}".format(self.stall_num, str(self.plate_nums)))
        if self.car_num == 5:
            outer_lap_pub.publish(True)

        plate_publisher.publish("funMode,passwd,{},{}".format(stall_order[self.car_num], "".join(self.plate_nums)))
        do_sift = False

class Plate_matcher():

    def __init__(self, path_array, blue_threshold):
        self.path_array = path_array
        self.car_num = 0
        self.get_template_from_path()
        self.bridge = CvBridge()
        self.blue_threshold = blue_threshold
        self.first_iter = True
        global do_sift
        self.last_frame_blue = False
        self.done_outside = False
        # self.model = models.load_model("../NN_character_recognition_blurred")

    def get_template_from_path(self):
        self.template = cv2.imread(self.path_array[self.car_num])
        print("using" + self.path_array[self.car_num])

    def read_camera(self, data):
        self.cam_img = self.bridge.imgmsg_to_cv2(data)
        blue = self.countBluePixels(self.cam_img)
        # if self.first_iter:
        #     self.time_last_blue = time.time()
        #     self.first_iter = False
        if blue > self.blue_threshold:
            if self.car_num == 5:
                outer_lap_pub.publish(True)
            # if time.time() - self.time_last_blue > TIME_TO_PASS_CAR:
            #     self.car_num += 1
            self.last_frame_blue = True
            # self.time_last_blue = time.time()
        # if self.countBluePixels(self.cam_img) > self.blue_threshold:
            print("attempting sift now")
            blue_pub.publish(True)
            self.blue = blue 
            global do_sift
            if do_sift:
                self.get_plate()
        elif self.last_frame_blue and blue < 8000:
            self.last_frame_blue = False
            do_sift = True
            self.car_num += 1
            if self.car_num == 6:
                self.done_outside = True
                # outer_lap_pub.publish(True)
            if self.car_num == 8:
                plate_publisher.publish("funMode,passwd,-1,XR58"
    def get_plate(self):
        self.get_template_from_path()
        gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(self.cam_img, cv2.COLOR_BGR2GRAY)
        height = np.shape(gray_template)[0]
        width = np.shape(gray_template)[1]
        


        sift = cv2.xfeatures2d.SIFT_create()

        gray_template = gray_template[:, :]
        gray_frame = gray_frame[:, :]

        kp1, desc1 = sift.detectAndCompute(gray_template, None)
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

            h, w = gray_template.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # out_img = cv2.drawMatches(self.template, kp1, self.cam_img, kp2, good_points,self.cam_img)
            # plt.imshow(out_img)
            # plt.show()

            # homography = cv2.polylines(self.cam_img, [np.int32(dst)], True, (255, 0, 0), 3)
            # plt.figure()
            # plt.imshow(homography)
            # plt.show()
            
            frame_pts = np.float32([[0,0],[0,width],[height,width], [height,0]])
            #check if transform is valid first, check that the 4 pts have right coords relative to each other
            corners = list(dst)
            # print (corners)
            c_pts = [pt[0] for pt in corners]
            
            if (c_pts[0][1] < c_pts[1][1] and c_pts[0][0] < c_pts[3][0] and
                 c_pts[3][1] < c_pts[2][1] and c_pts[1][0] < c_pts[2][0] and 
                 c_pts[0][0] < c_pts[2][0] and c_pts[3][1] < c_pts[1][1]): 
                print("transform success. requesting prediction")

                M = cv2.getPerspectiveTransform(dst, frame_pts)
                transformed = cv2.warpPerspective(self.cam_img, M, (height,width))
                # cv2.imwrite("../sim_plates/{}_plate{}.jpg".format(self.num_plates, self.blue), transformed)
                plate = Plate(transformed, self.car_num)
        
    def countBluePixels(self, img):
        if self.done_outside:
            img = img[:,int((1.0/2)*1280):]
        else:
            img = img[:, :1780/2]
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
    plate_matcher = Plate_matcher(paths, BLUE_THRESH)
    rospy.init_node('plate_matcher')
    rospy.Subscriber('R1/pi_camera/image_raw', ImageMsg, plate_matcher.read_camera)
    plate_publisher = rospy.Publisher('license_plate', String, queue_size=1)
    outer_lap_pub = rospy.Publisher('outer_lap_complete', Bool, queue_size=1)
    blue_pub = rospy.Publisher('blue_stop', Bool, queue_size=1)
    
    rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.spin()