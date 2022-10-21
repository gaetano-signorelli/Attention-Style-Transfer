'''
This layers combines features coming from different encoding levels.
Higher level features are down-sampled and then concatenated with the others.
'''

import tensorflow as tf
from tensorflow.keras import layers

class FeatureCombinatorLayer(layers.Layer):

    def __init__(self, encoded_shapes):

        super(FeatureCombinatorLayer, self).__init__()

        self.encoded_shapes = encoded_shapes

        self.conc_layer = layers.Concatenate()

    @tf.function
    def call(self, encoded_features):

        assert len(encoded_features)>2

        steps = len(encoded_features)

        shape = self.encoded_shapes[steps-1]
        down_sample_layer = layers.Resizing(height=shape[1], width=shape[2])

        x = down_sample_layer(encoded_features[0])
        for i in range(1, steps-1):
            x = self.conc_layer([x, down_sample_layer(encoded_features[i])])

        x = self.conc_layer([x, encoded_features[-1]])

        return x
