#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, DTParam, ParamType
from duckietown_msgs.msg import BoolStamped, ObstacleImageDetection, ObstacleType, Rect, ObstacleImageDetectionList
from sensor_msgs.msg import CompressedImage
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time
from PIL import Image


class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObstacleDetectionNode, self).__init__(
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

        # Publishers
        self.pub_detection_image = rospy.Publisher("~debug/detection_image/compressed", CompressedImage,
                                                   queue_size=1)
        self.pub_detection_flag = rospy.Publisher("~detection/flag", BoolStamped, queue_size=1)

        # DNN Configuration
        # TODO: Frank fix this shit to load from config!
        self.category_index = {1: {'id': 1, 'name': 'cone'},
                               2: {'id': 2, 'name': 'duckie'},
                               3: {'id': 3, 'name': 'duckiebot'}}
        self.model = "mobileNet-v2.tflite"
        self.detection_threashold = 0.1

        self.interpreter = Interpreter(self.model)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']

        self.log("Initialization completed.")

    def cb_image(self, image_msg):

        # Rate Limitation
        now = rospy.Time.now()
        if now - self.last_stamp < self.rate_limitation:
            return
        else:
            self.last_stamp = now

        # Setup Messages
        detection_flag_msg_out = BoolStamped()

        # Obtain Image
        image_cv = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")

        # DNN Inference
        # TODO: Implement DNN Inference
        image = Image.fromarray(image_cv).convert('RGB').resize(
            (self.input_width, self.input_height), Image.ANTIALIAS)
        (im_width, im_height) = image.size
        start_time = time.monotonic()
        results = self.detect_objects(self.interpreter, image, self.detection_threashold)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        print("Total time cost: {} ms".format(elapsed_ms))
        image_canvas = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        print(results)

        # Publish Detection Flag if obstacle is seen
        detection_flag_msg_out.header = image_msg.header
        detection_flag_msg_out.data = (len(results["score"]) > 0)
        self.pub_detection_flag.publish(detection_flag_msg_out)

        # Visualize detection
        if detection_flag_msg_out.data:
            # TODO: Draw Bounding Box here
            image_cv = None
            image_msg_out = self.bridge.cv2_to_compressed_imgmsg(image_cv)
            self.pub_detection_image.publish(image_msg_out)

    def set_input_tensor(self, interpreter, image):
        """Sets the input tensor."""
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, interpreter, index):
        """Returns the output tensor at the given index."""
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor

    def limit_bound(self, val):
        return min(max(0, val), 1)

    def detect_objects(self, interpreter, image, threshold):
        """Returns a list of detection results, each a dictionary of object info."""
        self.set_input_tensor(interpreter, image)
        interpreter.invoke()
        print(interpreter.get_output_details())
        # Get all output details
        boxes = self.get_output_tensor(interpreter, 0)
        classes = self.get_output_tensor(interpreter, 1)
        scores = self.get_output_tensor(interpreter, 2)
        count = int(self.get_output_tensor(interpreter, 3))
        (im_width, im_height) = image.size
        results = {'bounding_box': [], 'class_id': [], 'score': []}
        print(boxes)
        for i in range(count):
            if scores[i] >= threshold:
                results['bounding_box'].append([
                    self.limit_bound(boxes[i][0]) * im_height, self.limit_bound(boxes[i][1]) * im_width,
                    self.limit_bound(boxes[i][2]) * im_height, self.limit_bound(boxes[i][3]) * im_width])
                results['class_id'].append(classes[i].astype(int) + 1)
                results['score'].append(scores[i])
        results['bounding_box'] = np.array(results['bounding_box'])
        results['class_id'] = np.array(results['class_id'])
        results['score'] = np.array(results['score'])
        return results


if __name__ == '__main__':
    obstacle_detection_node = ObstacleDetectionNode('obstacle_detection')
    rospy.spin()
