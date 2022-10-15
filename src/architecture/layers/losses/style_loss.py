import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.losses.mse_loss import MSELossLayer

from src.architecture.config import *

class StyleLossLayer(layers.Layer):

    def __init__(self):

        super(StyleLossLayer, self).__init__()

        self.mse_layer = MSELossLayer()

    @tf.function
    def get_mean_std(self, features):

        mean = tf.math.reduce_mean(features, axis=(1,2), keepdims=True)
        #std = tf.math.reduce_std(features, axis=(1,2), keepdims=True)

        diff = tf.math.abs(features-mean)
        std = tf.math.reduce_mean(diff, axis=(1,2), keepdims=True)

        return mean, std

    @tf.function
    def style_loss(self, x, target):

        x_mean, x_std = self.get_mean_std(x)
        target_mean, target_std = self.get_mean_std(target)

        loss_mean = self.mse_layer([x_mean, target_mean])
        loss_std = self.mse_layer([x_std, target_std])

        style_loss = loss_mean + loss_std

        return style_loss

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        x_features = inputs[0]
        target_features = inputs[1]

        n_features = len(x_features)

        loss = self.style_loss(x_features[1], target_features[1])
        for i in range(2, n_features):
            loss += self.style_loss(x_features[i], target_features[i])

        return loss
