#! /usr/bin/env python
import rospy
import cv2
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

directory = '/home/fizzer/ros_ws/src/2020T1_competition/enph353/cnn-enph353/training_pictures'
bridge = CvBridge()
start_index_park = 0
start_index_ped = 0

def add_zeros(digit):
	length = len(str(digit))
	if length == 1:
		zeros = "000"
	if length == 2:
		zeros = "00"
	if length == 3:
		zeros = "0"

	return zeros + str(digit)

def image_name(char):
	global start_index_park
	global start_index_ped

	if char == "q":
		index = start_index_park
		start_index_park += 1
		return "park" + add_zeros(index) + "Y"
	elif char == "w":
		index = start_index_park
		start_index_park += 1
		return "park" + add_zeros(index) + "N"
	elif char == "e":
		index = start_index_ped
		start_index_ped += 1
		return "ped" + add_zeros(index) + "Y"
	elif char == "r":
		index = start_index_ped
		start_index_ped += 1
		return "ped" + add_zeros(index) + "N"

def image_callback(data):
	print("Recieved an image!")

	newPicture = raw_input("New Picture (reset each time with y, else n): ")

	if str(newPicture) == "y":
		return
	else:
		choice = raw_input('Legend:\n\nq: Park-Y\nw: Park-N\ne: Ped-Y\nr:Ped-N\n\nChoice:')
		name = image_name(choice)

		try:
			image = bridge.imgmsg_to_cv2(data)
			os.chdir(directory)
			cv2.imwrite(name + ".jpg", image)
			
		except CvBridgeError, e:
			print(e)

if __name__=="__main__":
    rospy.init_node('img_save', anonymous=True)
    rospy.Subscriber('/R1/pi_camera/image_raw', Image, image_callback)
    rospy.Rate(10)
    rospy.spin()