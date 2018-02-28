import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np


###################################################################################################
#                                                                                                 #
#                               Functions for Input generation                                    #
#                                                                                                 #
###################################################################################################
def read_im(im_path, im_shape=(227, 227, 3)):
    """
    Reading image

    :param im_path: str | path to the image
    :param im_shape: tuple | input size, default is (227, 227, 3)
    :return: array | image
    """
    im = (imread(im_path)[:, :, :3]).astype(np.float32)
    if im.shape != im_shape:
        im = imresize(im, im_shape)
    return im


def preprcess_im(im):
    """
    Preprocessing image : zero mean and RGB -> BGR

    :param im: array | array of RGB image
    :return: array | zero mean BGR image
    """
    im_ = im - np.mean(im)
    im_[:, :, 0], im_[:, :, 2] = im_[:, :, 2], im_[:, :, 0]
    return im_


def get_crop(im, idc, im_shape=(227, 227, 3)):
    """
    generates 9 crops from the original image

    :param im: array | array of RGB image
    :param idc: int | indice of the crop
    :param im_shape: tuple | input size, default is (227, 227, 3)
    :return: array | crop image
    """
    out = np.zeros((227, 227, 3))
    h = int(im_shape[0] / 3)
    out[((idc % 3) * h):(((idc % 3) + 1) * h), int(idc/3) * h:(int(idc/3) + 1) * h, :] = \
        im[((idc % 3) * h):(((idc % 3) + 1) * h), (int(idc/3) * h):((int(idc/3) + 1) * h), :]
    return out


###################################################################################################
#                                                                                                 #
#                             Functions to create the architecture                                #
#                                                                                                 #
###################################################################################################
def conv(input_data, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    c_i = input_data.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv_ = convolve(input_data, kernel)
    else:
        input_groups = tf.split(input_data, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv_ = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv_, biases), [-1] + conv_.get_shape().as_list()[1:])


###################################################################################################
#                                                                                                 #
#                             Functions to create the architecture                                #
#                                                                                                 #
###################################################################################################
def alexnet_arch(x, net_data, output_layer=8):
    """
    Architecture of AlexNet

    :param x: tensor | input
    :param net_data: dict | trained filters
    :param output_layer: int | layer number from which we compute the feature
    :return: output of the specified layer
    """
    outputs = []
    conv1_ = conv(x, tf.Variable(net_data["conv1"][0]), tf.Variable(net_data["conv1"][1]), 11, 11, 96, 4, 4,
                    padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_)
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    outputs.append(maxpool1)
    conv2_ = conv(maxpool1, tf.Variable(net_data["conv2"][0]), tf.Variable(net_data["conv2"][1]), 5, 5, 256, 1, 1,
                    padding="SAME", group=2)
    conv2 = tf.nn.relu(conv2_)
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    outputs.append(maxpool2)
    conv3_ = conv(maxpool2, tf.Variable(net_data["conv3"][0]), tf.Variable(net_data["conv3"][1]), 3, 3, 384, 1, 1,
                    padding="SAME", group=1)
    conv3 = tf.nn.relu(conv3_)
    outputs.append(conv3)
    conv4_ = conv(conv3, tf.Variable(net_data["conv4"][0]), tf.Variable(net_data["conv4"][1]), 3, 3, 384, 1, 1,
                    padding="SAME", group=2)
    conv4 = tf.nn.relu(conv4_)
    outputs.append(conv4)
    conv5_ = conv(conv4, tf.Variable(net_data["conv5"][0]), tf.Variable(net_data["conv5"][1]), 3, 3, 256, 1, 1,
                    padding="SAME", group=2)
    conv5 = tf.nn.relu(conv5_)
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    outputs.append(maxpool5)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]),
                           tf.Variable(net_data["fc6"][0]), tf.Variable(net_data["fc6"][1]))
    outputs.append(fc6)
    fc7 = tf.nn.relu_layer(fc6, tf.Variable(net_data["fc7"][0]), tf.Variable(net_data["fc7"][1]))
    outputs.append(fc7)
    fc8 = tf.nn.xw_plus_b(fc7, tf.Variable(net_data["fc8"][0]), tf.Variable(net_data["fc8"][1]))
    outputs.append(fc8)
    return outputs[output_layer - 1]
