'''
This layers normalizes (substract mean and divide by standard deviation) the input
along the specified axis.
'''

import tensorflow as tf
from tensorflow.keras import layers

class NormalizeLayer(layers.Layer):

    def __init__(self, axis):

        super(NormalizeLayer, self).__init__()

        self.axis=axis

    @tf.function
    def call(self, x):

        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        #std = tf.math.reduce_std(x, axis=self.axis, keepdims=True) + 1e-5

        var = tf.math.reduce_variance(x, axis=self.axis, keepdims=True)
        var = tf.math.maximum(var, 1e-9) #Necessary to avoid Nan by computing sqrt of small values
        std = tf.math.sqrt(var) + 1e-8

        norm_x = (x-mean) / std

        return norm_x
