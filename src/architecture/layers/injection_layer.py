import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.normalize_layer import NormalizeLayer

class InjectionLayer(layers.Layer):

    def __init__(self, encoded_shape):

        super(InjectionLayer, self).__init__()

        self.h = encoded_shape[1] #Height
        self.w = encoded_shape[2] #Width
        self.c = encoded_shape[3] #Channels

        self.norm_layer = NormalizeLayer()

        self.conv_content = layers.Conv2D(self.c, kernel_size=1)
        self.conv_style = layers.Conv2D(self.c, kernel_size=1)
        self.conv_output = layers.Conv2D(self.c, kernel_size=1)

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        content = inputs[0]
        style = inputs[1]

        content = self.conv_content(content)
        style = self.conv_style(style)

        mean_style = tf.math.reduce_mean(style, axis=(1,2), keepdims=True)
        std_style = tf.math.reduce_std(style, axis=(1,2), keepdims=True) + 1e-5

        norm_content = self.norm_layer(content)

        injection = (norm_content * std_style) + mean_style

        output = self.conv_output(injection)
        output = output + content

        return output
