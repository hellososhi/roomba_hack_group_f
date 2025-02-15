#!/usr/bin/env python3

from math import pi
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import actionlib
from actionlib_msgs.msg import GoalStatus
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import copy
import time


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

        # actionlib
        self.action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.action_client.wait_for_server()  # action serverの準備ができるまで待つ

        self.bridge = CvBridge()
        self.rgb_image = None

    def callback_rgbd(self, data1, data2, data3):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        # cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_image = cv_array

        self.camera_info = data3

    def set_goal(self, x, y, yaw):
        self.goal = MoveBaseGoal()  # goalのメッセージの定義
        self.goal.target_pose.header.frame_id = 'map'  # マップ座標系でのゴールとして設定
        self.goal.target_pose.header.stamp = rospy.Time.now()  # 現在時刻

        # ゴールの姿勢を指定
        self.goal.target_pose.pose.position.x = x
        self.goal.target_pose.pose.position.y = y
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)  # 回転はquartanionで記述するので変換
        self.goal.target_pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])

    def send_action(self, duration=30.0):
        self.action_client.send_goal(self.goal)  # ゴールを命令
        result = self.action_client.wait_for_result(rospy.Duration(duration))
        return result

    def process(self):
        image_path = "/root/roomba_hack/catkin_ws/src/three-dimensions_tutorial/images/"

        self.set_goal(3.5, 4.0, pi/2)
        self.action_client.send_goal(self.goal)

        while not rospy.is_shutdown() and self.action_client.get_state() != GoalStatus.SUCCEEDED:
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

            print(self.action_client.get_state())
        if self.action_client.get_state()==GoalStatus.SUCCEEDED:
            print(self.action_client.get_result())


if __name__ == '__main__':
    od = ObjectDetection()
    try:
        time.sleep(3)
        od.process()
    except rospy.ROSInitException:
        pass
