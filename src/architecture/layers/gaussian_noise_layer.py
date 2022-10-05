import tensorflow as tf
from tensorflow.keras import layers

class GaussianNoiseLayer(layers.Layer):

    def __init__(self):

        super(GaussianNoiseLayer, self).__init__()

    @tf.function
    def call(self, x):

        std = tf.random.uniform(shape=(), minval=0.01, maxval=0.02)
        input_shape = tf.shape(x)

        gaussian_noise = tf.random.normal(input_shape, mean=0.0, stddev=std)

        output = layers.Add()([x, gaussian_noise])

        return output
