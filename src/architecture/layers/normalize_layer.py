import tensorflow as tf
from tensorflow.keras import layers

class NormalizeLayer(layers.Layer):

    def __init__(self, axis):

        super(NormalizeLayer, self).__init__()

        self.axis=axis

    @tf.function
    def call(self, x):

        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        std = tf.math.reduce_std(x, axis=self.axis, keepdims=True) + 1e-5

        norm_x = (x-mean) / std

        return norm_x
