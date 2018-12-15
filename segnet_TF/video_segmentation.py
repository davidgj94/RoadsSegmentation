import tensorflow as tf
import flags; FLAGS = tf.app.flags.FLAGS
import numpy as np
from create_TFRecords import read_records
from model import segnet_extended
from score import compute_hist, print_hist_summary
import pdb
import cv2
from downscale import _downscale as downscale
from downscale import get_padding
import vis


palette = np.array([[255,255,255],[0, 0, 255],[0, 255, 0]])

def _video_segmentation(vidcap, out, delay_msec, x_pad, y_pad):

    with tf.Graph().as_default():

        image  = tf.placeholder(tf.float32, shape=[1, 544, 1024, 3], name="input")
        logits = segnet_extended(image)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session() as sess:

            sess.run(init_op)

            success = True
            time_msec = 0

            while success:
    
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(time_msec))
                success, img = vidcap.read()

                if success:

                    img_down = downscale(img, FLAGS.max_dim, False)
                    img_down = img_down.astype(np.float32)
                    img_down = img_down[np.newaxis,:]
                    _logits = sess.run(logits, feed_dict={image: img_down})
                    mask = np.argmax(np.squeeze(_logits), axis=-1)

                    mask = mask[y_pad[0]:-y_pad[-1], x_pad[0]:-x_pad[1]]
                    img_down = np.squeeze(img_down)
                    img_down = img_down[y_pad[0]:-y_pad[-1], x_pad[0]:-x_pad[1]]
                    vis_img = vis.vis_seg(np.squeeze(img_down[...,::-1]), mask, palette)
                    out.write(vis_img[...,::-1])

                    time_msec += delay_msec

                else:
                    out.release()

def main(args):

    vidcap = cv2.VideoCapture(FLAGS.video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, img = vidcap.read()
    img_aux = downscale(img, FLAGS.max_dim, False, pad_img=False)
    height, width = img_aux.shape[:2]
    x_pad = get_padding(width)
    y_pad = get_padding(height)
    out = cv2.VideoWriter(FLAGS.save_path_video, cv2.VideoWriter_fourcc(*'XVID'), FLAGS.fps, (width, height))
    delay_msec = int(1000 * (1 / FLAGS.fps))

    _video_segmentation(vidcap, out, delay_msec, x_pad, y_pad)

if __name__ == "__main__":
    tf.app.run() # wrapper that handles flags parsing.