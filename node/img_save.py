#! /usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image



if __name__=="__main__":
    rospy.init_node('img_save', anonymous=True)
    rospy.Subscriber('/R1/pi_camera/image_raw', Image, image_callback)
    rospy.rate(10)
    rospy.spin()