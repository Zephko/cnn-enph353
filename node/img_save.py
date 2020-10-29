#! /usr/bin/env python
import rospy
import cv2
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

directory = '/home/fizzer/ros_ws/src/2020T1_competition/enph353/cnn-enph353/training_pictures'
bridge = CvBridge()
start_index_park = 131
start_index_ped = 103

def add_zeros(digit):
	length = len(str(digit))
	if length == 1:
		zeros = "000"
	if length == 2:
		zeros = "00"
	if length == 3:
		zeros = "0"

	return zeros + str(digit)

def image_name(park_bool, ped_bool):
	global start_index_park
	global start_index_ped

	if park_bool == "y":
		park_index = start_index_park
		start_index_park += 1
		park_name = "park" + add_zeros(park_index) + "Y"

	elif park_bool == "n":
		park_index = start_index_park
		start_index_park += 1
		park_name = "park" + add_zeros(park_index) + "N"

	if ped_bool == "y":
		ped_index = start_index_ped
		start_index_ped += 1
		ped_name = "ped" + add_zeros(ped_index) + "Y"
	elif ped_bool == "n":
		ped_index = start_index_ped
		start_index_ped += 1
		ped_name = "ped" + add_zeros(ped_index) + "N"

	return [park_name, ped_name]

def image_callback(data):
	print("Recieved an image!")

	newPicture = raw_input("New Picture (reset each time with y, else n): ")

	if str(newPicture) == "y":
		return
	else:
		image = bridge.imgmsg_to_cv2(data)
		cv2.imshow('image', image)
		cv2.waitKey(10)



		park_bool = raw_input("parking? (y/n)")
		ped_bool = raw_input("pedestrian? (y/n)")
		
		potential_names = image_name(park_bool, ped_bool)

		for name in potential_names:
			try:
				
				os.chdir(directory)
				cv2.imwrite(name + ".jpg", image)
				
			except CvBridgeError, e:
				print(e)

if __name__=="__main__":
    rospy.init_node('img_save', anonymous=True)
    rospy.Subscriber('/R1/pi_camera/image_raw', Image, image_callback)
    rospy.Rate(500)
    rospy.spin()