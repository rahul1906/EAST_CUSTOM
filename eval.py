import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import json
from PIL import Image

import locality_aware_nms as nms_locality
# import lanms

tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './tmp/east_icdar2015_resnet_v1_50_rbox/', '')

from transform import four_point_transform
import pytesseract
import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS



def get_image(path):
    return cv2.imread(path)[:, :, ::-1]



def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    print('boxes')
    print(boxes[1])
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def get_scaled_image(image, scaling_factor):
    scaled_image = cv2.resize(image, (image.shape[1] * scaling_factor, image.shape[0] * scaling_factor), \
                          interpolation=cv2.INTER_LANCZOS4)
    return scaled_image

def create_formatted_points(line):
    line_split = line.split(',')
    point_list = []
    for i in range(2,9,2):
        point_list.append(tuple(line_split[i-2 : i]))
    return point_list

def get_ocr_output(im, score, geometry, ratio_h, ratio_w):


    boxes= detect(score_map=score, geo_map=geometry)

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    dict = {}
    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            line = '{},{},{},{},{},{},{},{}'.format(
                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
            )
            # print(line)
            formatted_points = create_formatted_points(line)
            pts = np.array(formatted_points, dtype = "float32")
            # print(pts)
            # apply the four point tranform to the image
            warped = four_point_transform(im, pts)
            scaled_image = get_scaled_image(warped, 10)
            temp_string = pytesseract.image_to_string(Image.fromarray(scaled_image), lang='eng', boxes=False, config='--psm 7').strip().encode('ascii', 'ignore')
            # print(str(pts) +" : "+ temp_string)
            # print(temp_string)
            dict[str(formatted_points)] = temp_string

    return dict    


def main(argv=None):

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            
            path = '/home/rahul/self/text_detection_east/EAST_mod/tmp/images/1.jpg'
            im = get_image(path)
            im_resized, (ratio_h, ratio_w) = resize_image(im)

            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})

            output_json = json.dumps(get_ocr_output(im, score, geometry, ratio_h, ratio_w))
            # print(output_json)
            


if __name__ == '__main__':
    tf.app.run()
