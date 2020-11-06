#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

path = '../training_pictures/plate_sift/plate_blue.png'

class Plate_matcher():

    def __init__(self, img_path):
        self.img_path =img_path
        self.get_img_from_path()
        self.bridge = CvBridge()

    def get_img_from_path(self):
        self.img = cv2.imread(self.img_path)

    def read_camera(self, data):
        self.cam_img = self.bridge.imgmsg_to_cv2(data)
        self.get_plate()


    def get_plate(self):

        #TODO run SIFT on the captured frame

        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.cvtColor(self.cam_img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(self.cam_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(gray_img, None)
        kp2, desc2 = sift.detectAndCompute(gray_frame, None)
        # img = cv2.drawKeypoints(gray_img, kp1, img)
        # frame = cv2.drawKeypoints(gray_frame, kp2, frame) 

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # matches = bf.match(desc1, desc2)
        # matches = sorted(matches, key = lambda x: x.distance)FLANN_INDEX_KDTREE = 1
        #find good matches
        good_points = []
        min_matches = 10
        for m, n in matches:
            if m.distance< 0.7 * n.distance:
                good_points.append(m)
  
        if len(good_points) > min_matches:
            query_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = gray_img.shape
            # h,w,d = img.shape
            # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # print(dst)
            out_img = cv2.drawMatches(self.img, kp1, self.cam_img, kp2, good_points,self.cam_img)
            # cv2.imshow("testing sift", out_img)
            plt.imshow(out_img)
            plt.show()

            homography = cv2.polylines(self.cam_img, [np.int32(dst)], True, (255, 0, 0), 3)
            plt.figure()
            plt.imshow(homography)
            plt.show()

            #extact red square from homography

            # corners = np.int32(dst)
            # print(corners)
            # print(corners[0][0][1])
            # print(corners[0][1], corners[1][1], corners[0][0], corners[1][0])
            # print(corners[0][0][1])
            # print(corners[1][1])
            # print(corners[0][0][0])
            # sq = homography[corners[0][0][1]:corners[1][1], corners[0][0][0]:corners[1][0]]
            # plt.figure()
            # plt.imshow(sq)
            # plt.show()

            # cv2.imshow("Homography", homography)

            # pixmap = self.convert_cv_to_pixmap(homography)
            # self.live_image_label.setPixmap(pixmap)

        else:
            out_img = cv2.drawMatches(self.img, kp1, self.cam_img, kp2, good_points,self.cam_img)
            # pixmap = self.convert_cv_to_pixmap(out_img)
            # self.live_image_label.setPixmap(pixmap)
            cv2.imshow("testing sift", out_img)

        # pixmap = self.convert_cv_to_pixmap(out_img)
        # self.live_image_label.setPixmap(pixmap)

if __name__ == "__main__":
    plate_matcher = Plate_matcher(path)
    rospy.init_node('plate_matcher')
    rospy.Subscriber('R1/pi_camera/image_raw', Image, plate_matcher.read_camera)
    rospy.Rate(10)
    rospy.spin()