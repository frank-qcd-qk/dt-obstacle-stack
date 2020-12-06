from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from utils import visualization_utils as viz_utils

category_index = {1: {'id': 1, 'name': 'cone'},
                  2: {'id': 2, 'name': 'duckie'},
                  3: {'id': 3, 'name': 'duckiebot'}}
(im_width, im_height) = (640, 480)


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def limit_bound(val):
    return min(max(0, val), 1)


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    print(interpreter.get_output_details())
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    (im_width, im_height) = image.size
    results = {'bounding_box': [], 'class_id': [], 'score': []}
    print(boxes)
    for i in range(count):
        if scores[i] >= threshold:
            results['bounding_box'].append([
                limit_bound(boxes[i][0]) * im_height, limit_bound(boxes[i][1]) * im_width,
                limit_bound(boxes[i][2]) * im_height, limit_bound(boxes[i][3]) * im_width])
            results['class_id'].append(classes[i].astype(int) + 1)
            results['score'].append(scores[i])
    results['bounding_box'] = np.array(results['bounding_box'])
    results['class_id'] = np.array(results['class_id'])
    results['score'] = np.array(results['score'])
    return results


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.1)
    args = parser.parse_args()

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    image = Image.open("image.png").convert('RGB').resize(
        (input_width, input_height), Image.ANTIALIAS)
    (im_width, im_height) = image.size
    start_time = time.monotonic()
    results = detect_objects(interpreter, image, args.threshold)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    print("Total time cost: {} ms".format(elapsed_ms))
    image_canvas = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    print(results)
    todraw = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_canvas,
        results['bounding_box'],
        results['class_id'],
        results['score'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
    )
    plt.imshow(todraw)
    plt.savefig('image_with_detection.png')


if __name__ == '__main__':
    main()
