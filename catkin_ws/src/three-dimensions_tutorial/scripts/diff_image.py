#!/usr/bin/env python3

from math import pi
import rospy
import tf
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from pytorchyolo import detect, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import copy
import time


class DetectionDistance:
    def __init__(self):
        rospy.init_node('detection_distance', anonymous=True)

        # Publisher
        self.detection_result_pub = rospy.Publisher('/detection_result', Image, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        odom_sub = rospy.Subscriber('/odom', Odometry, self.callback_odom)
        message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 1.0).registerCallback(self.callback_rgbd)

        self.bridge = CvBridge()
        self.rgb_image, self.depth_image = None, None

    def callback_rgbd(self, data1, data2):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_image = cv_array

    def callback_odom(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.yaw = self.get_yaw_from_quaternion(data.pose.pose.orientation)

    def go_straight(self, dis, velocity=0.3):
        vel = Twist()
        x0 = self.x
        y0 = self.y
        while (np.sqrt((self.x-x0)**2+(self.y-y0)**2) < dis):
            vel.linear.x = velocity
            vel.angular.z = 0.0
            self.cmd_vel_pub.publish(vel)
            rospy.sleep(0.1)
        self.stop()

    def turn(self, yaw, yawrate):
        vel = Twist()
        yaw0 = self.yaw
        while (abs(self.yaw-yaw0) < np.deg2rad(yaw)):
            vel.linear.x = 0.0
            vel.angular.z = yawrate
            self.cmd_vel_pub.publish(vel)
            rospy.sleep(0.1)
        self.stop()

    def stop(self):
        vel = Twist()
        vel.linear.x = 0.0
        vel.angular.z = 0.0
        self.cmd_vel_pub.publish(vel)

    def get_yaw_from_quaternion(self, quaternion):
        e = tf.transformations.euler_from_quaternion(
            (quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return e[2]

    def process(self):
        image_path = "/root/roomba_hack/catkin_ws/src/three-dimensions_tutorial/images/"
        path = "/root/roomba_hack/catkin_ws/src/three-dimensions_tutorial/yolov3/"

        # load category
        with open(path+"data/coco.names") as f:
            category = f.read().splitlines()

        # prepare model
        model = models.load_model(path+"config/yolov3.cfg", path+"weights/yolov3.weights")

        x1, y1, x2, y2 = (0, 0, 0, 0)
        start_time = float('inf')
        pre_image = None
        is_found_person = False
        sum_diff = 0
        while not rospy.is_shutdown() and time.time()-start_time < 10:
            if self.rgb_image is None:
                continue
            # inference
            rgb_image = copy.copy(self.rgb_image)
            boxes = detect.detect_image(model, rgb_image)
            # [[x1, y1, x2, y2, confidence, class]]
            for box in boxes:
                cls_pred = category[int(box[5])]
                if cls_pred == 'person':
                    if is_found_person:
                        trimed_image = rgb_image[y1:y2, x1:x2, :]
                        diff_image = cv2.absdiff(trimed_image, pre_image)
                        diff_gray_image = cv2.cvtColor(diff_image, cv2.COLOR_RGB2GRAY)
                        ret, diff_bin_image = cv2.threshold(diff_gray_image, 20, 255, cv2.THRESH_BINARY)
                        cv2.imwrite(image_path+'pre.png', cv2.cvtColor(pre_image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(image_path+'trimmed.png', cv2.cvtColor(trimed_image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(image_path+"diff.png", diff_bin_image)
                        sum_diff += diff_bin_image.sum()
                        pre_image = trimed_image
                    else:
                        is_found_person = True
                        print('found person')
                        x1, y1, x2, y2 = map(int, box[:4])
                        start_time = time.time()
                        pre_image = rgb_image[y1:y2, x1:x2, :]
                        print(pre_image.shape)

            # publish image
            result_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            detection_result = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.detection_result_pub.publish(detection_result)
            self.rgb_image = None

        print(sum_diff)
        if sum_diff > 1500000:
            cx = (x1+x2)/2
            if cx > rgb_image.shape[1]/2:
                self.turn(5, -0.5)
            else:
                self.turn(5, 0.5)
        else:
            cx = (x1+x2)/2
            if cx > rgb_image.shape[1]/2:
                self.turn(5, 0.5)
            else:
                self.turn(5, -0.5)
        self.go_straight(2.5)


if __name__ == '__main__':
    dd = DetectionDistance()
    try:
        dd.process()
    except rospy.ROSInitException:
        pass
