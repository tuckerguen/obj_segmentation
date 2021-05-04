#! /home/tucker/anaconda3/envs/detectron/bin/python

import rospy
from detectron2.utils.logger import setup_logger
import sys
import cv2
from ObjectSegmenter import ObjectSegmenter

setup_logger()

def main(args):
    ic = ObjectSegmenter()
    print('Initialized object segmentation object')
    rospy.init_node('obj_segmentation', anonymous=True)
    try:
        rate = rospy.Rate(1)  # ROS Rate at 5Hz
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
