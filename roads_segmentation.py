import utils
from lane_segmentation import segment_lanes
from marks_grouping import group_marks
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from matplotlib.patches import Circle
from skimage import measure
import sys
from math import ceil
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
import vis
from skimage.morphology import remove_small_objects
import tensorflow as tf
import segnet_TF
from PIL import Image
import argparse
import os.path
import cv2
import os
import shutil


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords', type=float, required=True, nargs=2)
    parser.add_argument('--save_dir', type=str)
    return parser


if __name__ == "__main__":

    args = make_parser().parse_args()

    img, mask_download = utils.download_img_mask(tuple(args.coords))
    mask_download = mask_download[:1200,:]
    img = img[:1200,:]

    skeleton = morphology.medial_axis(mask_download == 255)

    skeleton[0,:] = 0
    skeleton[-1,:] = 0
    skeleton[:,0] = 0
    skeleton[:,-1] = 0

    skleton_sections = utils.divide_skeleton(skeleton, img)

    crops = []
    H_list = []
    W_list = []
    mid_points = []
    angles = []
    one_section_segmentation = True

    for section in skleton_sections:

        section_mid_points, section_angles, section_height, _angles = utils.divide_section(section, M=200)

        if len(section_mid_points) > 1:
            one_section_segmentation = False

        for index, _ in enumerate(section_mid_points):

            crop = utils.extract_straight_section(img, section_mid_points[index], section_angles[index], section_height[index])
            H, W = crop.shape[:2]

            H_list.append(H)
            W_list.append(W)
            crops.append(crop)
            mid_points.append(section_mid_points[index])
            angles.append(section_angles[index])

    H_max = max(H_list)
    W_max = max(W_list)

    H_pad = H_max + int(ceil(float(H_max) / 32) * 32 - H_max)
    W_pad = W_max + int(ceil(float(W_max) / 32) * 32 - W_max)

    if one_section_segmentation:
        i_class = 0

    with tf.Graph().as_default():

        image  = tf.placeholder(tf.float32, shape=[1, H_pad, W_pad, 3], name="input")
        logits = segnet_TF.segnet_extended(image)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session() as sess:

            sess.run(init_op)

            if one_section_segmentation:
                total_mask = np.zeros(skeleton.shape, dtype=np.uint8)
            else:
                total_mask = np.zeros(skeleton.shape).astype(bool)

            for crop, mid_point, angle in zip(crops, mid_points, angles):

                padded_crop, x_pad, y_pad = utils.pad_img(crop, (H_pad, W_pad))
                padded_crop_ = padded_crop[...,::-1]
                padded_crop_ = padded_crop_[np.newaxis,:]
                padded_crop_ = padded_crop_.astype(np.float32)

                _logits = sess.run(logits, feed_dict={image: padded_crop_})

                mask = np.argmax(np.squeeze(_logits), axis=-1)
                mask = mask[y_pad[0]:H_pad-y_pad[-1], x_pad[0]:W_pad-x_pad[1]]

                if one_section_segmentation:
                
                    mask_lanes, num_lanes = segment_lanes(mask)

                    for i_lane in range(1, num_lanes+1):

                        i_class += 1
                        total_mask_ = utils.paste_mask(np.zeros(total_mask.shape).astype(bool), (mask_lanes == i_lane), mid_point, angle)
                        total_mask_ = binary_fill_holes(total_mask_.astype(int))
                        total_mask[total_mask_] = i_class

                else:

                    total_mask = utils.paste_mask(total_mask, (mask == 1), mid_point, angle)


            if one_section_segmentation:
                vis_img = vis.vis_seg(img, total_mask, vis.make_palette(i_class+1))
            else:
                total_mask = binary_fill_holes(total_mask.astype(int)).astype(int)
                total_mask = remove_small_objects(measure.label(total_mask, connectivity=2), min_size=50)
                new_total_mask, num_lanes = group_marks(total_mask)
                vis_img = vis.vis_seg(img, new_total_mask, vis.make_palette(num_lanes+1))

            if args.save_dir:
                new_save_dir = os.path.join(args.save_dir,"{}_{}".format(*tuple(args.coords)))
                if os.path.exists(new_save_dir):
                    shutil.rmtree(new_save_dir, ignore_errors=True)
                os.mkdir(new_save_dir)
                cv2.imwrite(os.path.join(new_save_dir, "mask.png"), total_mask)
                cv2.imwrite(os.path.join(new_save_dir, "img.png"), img[...,::-1])
                cv2.imwrite(os.path.join(new_save_dir, "vis_img.png"), vis_img[...,::-1])
            else:
                plt.imshow(vis_img)
                plt.show()


