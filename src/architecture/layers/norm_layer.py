import tensorflow as tf
from tensorflow.keras import layers

class NormalizeLayer(layers.Layer):

    def __init__(self, axis, eps=1e-5):

        super(NormalizeLayer, self).__init__()

        self.axis = axis
        self.eps = eps

    @tf.function
    def call(self, x):

        x_mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        x_std = tf.reduce_std(x, axis=self.axis, keepdims=True) + self.eps

        x_norm = (x - x_mean) / x_std

        return x_norm
