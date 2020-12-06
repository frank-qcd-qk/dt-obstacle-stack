#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, DTParam, ParamType
from duckietown_msgs.msg import BoolStamped, ObstacleImageDetection, ObstacleType, Rect, ObstacleImageDetectionList
from sensor_msgs.msg import CompressedImage
import numpy as np
import time


class ObstacleTrackingNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObstacleTrackingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Initialize the parameters
        self.process_frequency = DTParam('~process_frequency',
                                         param_type=ParamType.FLOAT)

        self.bridge = CvBridge()

        self.last_stamp = rospy.Time.now()

        self.rate_limitation = rospy.Duration.from_sec(1.0 / self.process_frequency.value)

        # Subscriber
        self.sub_image = rospy.Subscriber("~image", CompressedImage, self.cb_image, queue_size=1)
        self.sub_dnn = rospy.Subscriber("~dnn_bbox", ObstacleImageDetectionList, self.cb_detection, queue_size=1)

        # Publishers
        self.pub_detection_image = rospy.Publisher("~debug/tracking_image/compressed", CompressedImage,
                                                   queue_size=1)

        self.pub_bounding_box = rospy.Publisher("bbox2D", ObstacleImageDetectionList, queue_size=1)

        self.base_image = None

        self.latest_bbox = None

        self.trackers = None

        self.tracker_order = []

        self.tracker_type = "mosse"

        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        self.log("Initialization completed.")

    def update_bboxes(self, success, boxes, og_msg):
        for (asuccess, abox, adetection) in zip(success, boxes, self.tracker_order):
            if not asuccess:
                pass
            else:
                pass

    def cb_image(self, image_msg):
        self.base_image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        if self.latest_bbox is None or len(self.latest_bbox) == 0:
            image_msg_out = self.bridge.cv2_to_compressed_imgmsg(self.base_image)
            self.pub_detection_image.publish(image_msg_out)
            return
        else:
            success, boxes = self.trackers.update(self.base_image)

            new_detection = ObstacleImageDetectionList

            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                # TODO: Fix drawing bbox
                # cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

                # TODO: Publish Image here;;;
            return

    def cb_detection(self, detection_msg):
        self.latest_bbox = detection_msg.list
        self.trackers = cv2.MultiTracker_create()
        for detection in detection_msg.list:
            self.tracker_order.append(detection.type)
            local_bbox = [detection.rect.x, detection.rect.y, detection.rect.w, detection.rect.h]
            self.trackers.add(self.OPENCV_OBJECT_TRACKERS[self.tracker_type], self.base_image, local_bbox)
        return


if __name__ == '__main__':
    obstacle_tracking_node = ObstacleTrackingNode('obstacle_tracking')
    rospy.spin()
