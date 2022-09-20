#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pytorchyolo import detect, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import copy


class DetectionDistance:
    def __init__(self):
        rospy.init_node('detection_distance', anonymous=True)

        # Publisher
        self.detection_result_pub = rospy.Publisher('/detection_result', Image, queue_size=10)

        # Subscriber
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 1.0).registerCallback(self.callback_rgbd)

        self.bridge = CvBridge()
        self.rgb_image, self.depth_image = None, None

    def callback_rgbd(self, data1, data2):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_image = cv_array

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_image = cv_array

    def process(self):
        image_path = "/root/roomba_hack/catkin_ws/src/three-dimensions_tutorial/images/"
        path = "/root/roomba_hack/catkin_ws/src/three-dimensions_tutorial/yolov3/"

        # load category
        with open(path+"data/coco.names") as f:
            category = f.read().splitlines()

        # prepare model
        model = models.load_model(path+"config/yolov3.cfg", path+"weights/yolov3.weights")

        x1, y1, x2, y2 = (0, 0, 0, 0)
        pre_image = None
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                continue

            is_found_person = False
            # inference
            tmp_image = copy.copy(self.rgb_image)
            boxes = detect.detect_image(model, tmp_image)
            # [[x1, y1, x2, y2, confidence, class]]
            for box in boxes:
                cls_pred = category[int(box[5])]
                if cls_pred == "person":
                    if is_found_person:
                        trimed_image = tmp_image[y1:y2, x1:x2, :]
                        if pre_image:
                            diff_image = cv2.absdiff(trimed_image, pre_image)
                            diff_gray_image = cv2.cvtColor(diff_image, cv2.COLOR_RGB2GRAY)
                            diff_bin_image = diff_gray_image > 20
                            cv2.imwrite(image_path+"diff.png", diff_bin_image)
                        pre_image = trimed_image
                    else:
                        is_found_person = True
                        x1, y1, x2, y2 = map(int, box[:4])

            # plot bouding box
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cls_pred = int(box[5])
                tmp_image = cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                tmp_image = cv2.putText(tmp_image, category[cls_pred], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                print(category[cls_pred], self.depth_image[cy][cx]/1000, "m")

            # publish image
            tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2BGR)
            detection_result = self.bridge.cv2_to_imgmsg(tmp_image, "bgr8")
            self.detection_result_pub.publish(detection_result)


if __name__ == '__main__':
    dd = DetectionDistance()
    try:
        dd.process()
    except rospy.ROSInitException:
        pass
