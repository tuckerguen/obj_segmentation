#! /home/tucker/anaconda3/envs/detectron/bin/python

import rospy
from cv2_to_imgmsg import cv2_to_imgmsg
from detectron2.utils.logger import setup_logger
import sys
from sensor_msgs.msg import Image
from ObjectSegmenter import ObjectSegmenter

setup_logger()


class SegmentationNode:
    def __init__(self, visualize=False):
        self.image_pub = rospy.Publisher('/segmented_img', Image, queue_size=1)
        self.image_sub = rospy.Subscriber('/cameras/head_camera/image', Image, self.im_callback)
        self.objsegmenter = ObjectSegmenter()
        self.visualize = visualize

    def im_callback(self, data):
        outputs, im = self.objsegmenter.segment(data, y_crop_idx=600)
        labels = self.objsegmenter.get_labels(outputs)
        print(labels)
        print(self.objsegmenter.get_boxes(outputs))
        if self.visualize:
            out = self.objsegmenter.draw_output(outputs, im)
            # Publish segmented image
            self.image_pub.publish(cv2_to_imgmsg(out.get_image(), 'bgr8'))


if __name__ == '__main__':
    seg_node = SegmentationNode()
    rospy.init_node('obj_segmentation', anonymous=True)
    try:
        rate = rospy.Rate(1)  # ROS Rate at 5Hz
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
