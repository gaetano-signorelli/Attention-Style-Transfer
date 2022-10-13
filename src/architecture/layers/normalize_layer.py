import tensorflow as tf
from tensorflow.keras import layers

class NormalizeLayer(layers.Layer):

    def __init__(self):

        super(NormalizeLayer, self).__init__()

    @tf.function
    def call(self, x):

        mean = tf.math.reduce_mean(x, axis=(1,2), keepdims=True)
        std = tf.math.reduce_std(x, axis=(1,2), keepdims=True) + 1e-5

        norm_x = (x-mean) / std

        return norm_x
