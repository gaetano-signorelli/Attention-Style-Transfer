'''
This layers computes the content loss between encoded features of two images
'''

import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.losses.mse_loss import MSELossLayer
from src.architecture.layers.normalize_layer import NormalizeLayer

class ContentLossLayer(layers.Layer):

    def __init__(self):

        super(ContentLossLayer, self).__init__()

        self.norm_layer = NormalizeLayer(axis=(1,2))
        self.mse_layer = MSELossLayer()

    @tf.function
    def content_loss(self, x, target):

        #x = self.norm_layer(x)
        #target = self.norm_layer(x)

        content_loss = self.mse_layer([x, target])

        return content_loss

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        x_features = inputs[0]
        target_features = inputs[1]

        n_features = len(x_features)

        loss = self.content_loss(x_features[0], target_features[0])
        for i in range(1, n_features):
            loss += self.content_loss(x_features[i], target_features[i])

        return loss
