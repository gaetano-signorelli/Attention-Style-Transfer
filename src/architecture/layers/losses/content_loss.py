import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.losses.mse_loss import MSELossLayer
from src.architecture.layers.normalize_layer import NormalizeLayer

class ContentLossLayer(layers.Layer):

    def __init__(self):

        super(ContentLossLayer, self).__init__()

        self.norm_layer = NormalizeLayer()
        self.mse_layer = MSELossLayer()

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        x = inputs[0]
        target = inputs[1]

        #x = self.norm_layer(x)
        #target = self.norm_layer(target)

        loss = self.mse_layer([x, target])

        return loss
