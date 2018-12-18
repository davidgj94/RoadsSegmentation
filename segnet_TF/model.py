import tensorflow as tf
import pickle
from layers import unpool_with_argmax, conv_classifier, conv_layer_with_bn_caffe
import os
import os.path

def get_caffe_weights():
    caffe_weights_path = os.path.join(os.getcwd(),'segnet_TF','caffe_weights.pickle')
    with open(caffe_weights_path, 'rb') as handle:
        caffe_weights = pickle.load(handle)
    return caffe_weights

def segnet_extended(images, is_training=False):

    caffe_weights = get_caffe_weights()
    img_d = images.get_shape().as_list()[3]
    conv1_1 = conv_layer_with_bn_caffe(caffe_weights, images, [3, 3, img_d, 64], is_training, name="conv1_1")
    conv1_2 = conv_layer_with_bn_caffe(caffe_weights, conv1_1, [3, 3, 64, 64], is_training, name="conv1_2")
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    conv2_1 = conv_layer_with_bn_caffe(caffe_weights, pool1, [3, 3, 64, 128], is_training, name="conv2_1")
    conv2_2 = conv_layer_with_bn_caffe(caffe_weights, conv2_1, [3, 3, 128, 128], is_training, name="conv2_2")
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv3_1 = conv_layer_with_bn_caffe(caffe_weights, pool2, [3, 3, 128, 256], is_training, name="conv3_1")
    conv3_2 = conv_layer_with_bn_caffe(caffe_weights, conv3_1, [3, 3, 256, 256], is_training, name="conv3_2")
    conv3_3 = conv_layer_with_bn_caffe(caffe_weights, conv3_2, [3, 3, 256, 256], is_training, name="conv3_3")
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv4_1 = conv_layer_with_bn_caffe(caffe_weights, pool3, [3, 3, 256, 512], is_training, name="conv4_1")
    conv4_2 = conv_layer_with_bn_caffe(caffe_weights, conv4_1, [3, 3, 512, 512], is_training, name="conv4_2")
    conv4_3 = conv_layer_with_bn_caffe(caffe_weights, conv4_2, [3, 3, 512, 512], is_training, name="conv4_3")
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    
    conv5_1 = conv_layer_with_bn_caffe(caffe_weights, pool4, [3, 3, 512, 512], is_training, name="conv5_1")
    conv5_2 = conv_layer_with_bn_caffe(caffe_weights, conv5_1, [3, 3, 512, 512], is_training, name="conv5_2")
    conv5_3 = conv_layer_with_bn_caffe(caffe_weights, conv5_2, [3, 3, 512, 512], is_training, name="conv5_3")
    pool5, pool5_indices = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1, 2, 2, 1],
                                                        strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    """ End of encoder """

    """ Start decoder """
    unpool_5 = unpool_with_argmax(pool5, ind=pool5_indices, name="unpool_5")
    conv_decode5_3 = conv_layer_with_bn_caffe(caffe_weights, unpool_5, [3, 3, 512, 512], is_training, name="conv5_3_D")
    conv_decode5_2 = conv_layer_with_bn_caffe(caffe_weights, conv_decode5_3, [3, 3, 512, 512], is_training, name="conv5_2_D")
    conv_decode5_1 = conv_layer_with_bn_caffe(caffe_weights, conv_decode5_2, [3, 3, 512, 512], is_training, name="conv5_1_D")

    unpool_4 = unpool_with_argmax(conv_decode5_1, ind=pool4_indices, name="unpool_4")
    conv_decode4_3 = conv_layer_with_bn_caffe(caffe_weights, unpool_4, [3, 3, 512, 512], is_training, name="conv4_3_D")
    conv_decode4_2 = conv_layer_with_bn_caffe(caffe_weights, conv_decode4_3, [3, 3, 512, 512], is_training, name="conv4_2_D")
    conv_decode4_1 = conv_layer_with_bn_caffe(caffe_weights, conv_decode4_2, [3, 3, 512, 256], is_training, name="conv4_1_D")

    unpool_3 = unpool_with_argmax(conv_decode4_1, ind=pool3_indices, name="unpool_3")
    conv_decode3_3 = conv_layer_with_bn_caffe(caffe_weights, unpool_3, [3, 3, 256, 256], is_training, name="conv3_3_D")
    conv_decode3_2 = conv_layer_with_bn_caffe(caffe_weights, conv_decode3_3, [3, 3, 256, 256], is_training, name="conv3_2_D")
    conv_decode3_1 = conv_layer_with_bn_caffe(caffe_weights, conv_decode3_2, [3, 3, 256, 128], is_training, name="conv3_1_D")

    unpool_2 = unpool_with_argmax(conv_decode3_1, ind=pool2_indices, name="unpool_2")
    conv_decode2_2 = conv_layer_with_bn_caffe(caffe_weights, unpool_2, [3, 3, 128, 128], is_training, name="conv2_2_D")
    conv_decode2_1 = conv_layer_with_bn_caffe(caffe_weights, conv_decode2_2, [3, 3, 128, 64], is_training, name="conv2_1_D")

    unpool_1 = unpool_with_argmax(conv_decode2_1, ind=pool1_indices, name="unpool_1")
    conv_decode1_2 = conv_layer_with_bn_caffe(caffe_weights, unpool_1, [3, 3, 64, 64], is_training, name="conv1_2_D")
    logits = conv_classifier(caffe_weights, conv_decode1_2, name="conv1_1_D_roads")
    """ End of decoder """
        
    return logits