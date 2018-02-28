import tensorflow as tf
import numpy as np

import BBox.bbtools as t
from BBox.caffe_classes import class_names


class AlexNetClassifier():
    """
    This is an interface listing methods that are implemented when classifying an image using AlexNet

    """
    def __init__(self, im_path, model_path, im_shape=(227, 227, 3)):
        """
        Init the AlexNetClassifier object

        :param im_path: str | path to the input image
        :param model_path: str | path to the trained model
        :param im_shape: tuple | input size, default is (227, 227, 3)
        """
        self.im = t.read_im(im_path)
        self.shape = im_shape
        self.net_data = np.load(open(model_path, "rb"), encoding="latin1").item()
        self.im_crop = {'im_crop' + str(i + 1): t.get_crop(self.im, i, im_shape=im_shape) for i in range(9)}

    def get_feature(self, input_data, output_layer=8):
        """
        Generates AlexNet features

        :param input_data: list | list of arrays, each array represents a RGB image
        :param output_layer: int | layer number from which we compute the feature
        :return: list | list of the features
        """
        pre_data = [t.preprcess_im(el) for el in input_data]
        x = tf.placeholder(tf.float32, (None,) + self.shape)
        feat = t.alexnet_arch(x, self.net_data, output_layer=output_layer)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        output = sess.run(feat, feed_dict={x: pre_data})
        sess.close()
        return output

    def predict(self, input_data):
        """
        This function computes the classification

        :param input_data: list | list of arrays, each array represents a preprocessed BGR image
        :return: list | list of the last layer's features
        """
        x = tf.placeholder(tf.float32, (None,) + self.shape)
        feat = t.alexnet_arch(x, self.net_data, output_layer=8)
        prob = tf.nn.softmax(feat)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        output = sess.run(prob, feed_dict={x: input_data})
        sess.close()
        return output

    def print_class(self, input_data):
        """
        This function classify the input_data and prints its output label

        :param input_data: list | list of arrays, each array represents a RGB image
        """
        pre_data = [t.preprcess_im(el) for el in input_data]
        output = self.predict(pre_data)
        for input_im_ind in range(output.shape[0]):
            inds = np.argsort(output)[input_im_ind, :]
            print("Image", input_im_ind)
            print(class_names[inds[-1]], output[input_im_ind, inds[-1]])

    def nearest_crop(self, layer=8):
        """
        This function returns the nearest crop of the original image using the features of layer

        :param layer: int | layer number from which we compute the feature
        :return: tuple | returns the lowest distance and the crop id
        """
        im_feat = self.get_feature([self.im], output_layer=layer)
        key_list = list(self.im_crop.keys())
        crop_list = [self.im_crop[key] for key in key_list]
        crop_feat = self.get_feature(crop_list, output_layer=layer)
        crop_feat = [el for el in crop_feat]
        dist = [np.linalg.norm(im_feat - el) for el in crop_feat]
        nearest = dist.index(min(dist))
        return min(dist), key_list[nearest]
