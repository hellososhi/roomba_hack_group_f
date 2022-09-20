#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import copy


class ObjectDetection:
    def __init__(self):
        rospy.init_node('object_detection', anonymous=True)

        # Publisher
        self.detection_result_pub = rospy.Publisher('/detection_result', Image, queue_size=10)
        self.masked_depth_pub = rospy.Publisher('/masked_depth/image', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('/masked_depth/camera_info', CameraInfo, queue_size=10)

        # Subscriber
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        cam_info = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, cam_info], 10, 1.0).registerCallback(self.callback_rgbd)

        self.bridge = CvBridge()
        self.rgb_image = None

    def callback_rgbd(self, data1, data2, data3):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        # cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_image = cv_array

        self.camera_info = data3

    def process(self):
        image_path = "/root/roomba_hack/catkin_ws/src/three-dimensions_tutorial/images/"

        while not rospy.is_shutdown():
            if self.rgb_image is None:
                continue
            img_rgb = copy.copy(self.rgb_image)
            img_depth = copy.copy(self.depth_image)
            camera_info = copy.copy(self.camera_info)
            
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
            img_mask = cv2.inRange(img_hsv, (0, 100, 100), (255, 255, 255))
            img_bin = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours, hierarchy = cv2.findContours(255-img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_contours = cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 3)
            
            # publish image
            # img_result = cv2.cvtColor(img_contours, cv2.COLOR_RGB2BGR)
            img_result = img_contours
            detection_result = self.bridge.cv2_to_imgmsg(img_result, "bgr8")
            self.detection_result_pub.publish(detection_result)
            self.rgb_image = None

            masked_depth = np.where(img_mask, img_depth, 0)
            masked_depth = self.bridge.cv2_to_imgmsg(masked_depth, "passthrough")
            masked_depth.header = camera_info.header
            self.masked_depth_pub.publish(masked_depth)
            self.camera_info_pub.publish(camera_info)


if __name__ == '__main__':
    od = ObjectDetection()
    try:
        od.process()
    except rospy.ROSInitException:
        pass
