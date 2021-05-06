#! /home/tucker/anaconda3/envs/detectron/bin/python

import rospy
import json
import numpy as np
from cv2_to_imgmsg import cv2_to_imgmsg
from detectron2.utils.logger import setup_logger
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ObjectSegmenter import ObjectSegmenter
from segment_msg.msg import Segmentation

setup_logger()


class SegmentationNode:
    def __init__(self, visualize=False):
        self.image_pub = rospy.Publisher('/segmented_img', Image, queue_size=1)
        self.image_sub = rospy.Subscriber('/cameras/head_camera/image', Image, self.im_callback)
        self.segment_pub = rospy.Publisher('/segmentations', Segmentation, queue_size=1)
        self.objsegmenter = ObjectSegmenter()
        self.visualize = visualize

    def im_callback(self, data):
        outputs, im = self.objsegmenter.segment(data, y_crop_idx=700)
        labels = self.objsegmenter.get_labels(outputs)
        box_list = np.asarray(self.objsegmenter.get_boxes(outputs).tensor.tolist())
        if self.visualize:
            out = self.objsegmenter.draw_output(outputs, im)
            # Publish segmented image
            self.image_pub.publish(cv2_to_imgmsg(out.get_image(), 'bgr8'))

        # Publish segmentation data as Segmentation message
        seg_msg = Segmentation()
        # Add labels
        seg_msg.classes = labels
        # Flatten data into 1d array
        seg_msg.boxes.data = box_list.flatten()
        # Populate layout info
        dims = box_list.shape
        seg_msg.boxes.layout.dim[0].label = 'classes'
        seg_msg.boxes.layout.dim[0].size = dims[0]
        seg_msg.boxes.layout.dim[0].stride = dims[0]*dims[1]
        seg_msg.boxes.layout.dim[1].label = 'bbox coordinates'
        seg_msg.boxes.layout.dim[1].size = dims[1]
        seg_msg.boxes.layout.dim[1].stride = dims[1]
        self.segment_pub.publish(seg_msg)


if __name__ == '__main__':
    seg_node = SegmentationNode(visualize=True)
    rospy.init_node('obj_segmentation', anonymous=True)
    try:
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
