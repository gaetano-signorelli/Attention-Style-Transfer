import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.losses.mse_loss import MSELossLayer

class IdentityLossLayer(layers.Layer):

    def __init__(self):

        super(IdentityLossLayer, self).__init__()

        self.mse_layer = MSELossLayer()

    @tf.function
    def call(self, inputs):

        assert len(inputs)==4

        x_image = inputs[0]
        target_image = inputs[1]
        x_features = inputs[2]
        target_features = inputs[3]

        n_features = len(x_features)

        image_loss = self.mse_layer([x_image, target_image])

        features_loss = self.mse_layer([x_features[0], target_features[0]])
        for i in range(1, n_features):
            loss += self.mse_layer([x_features[i], target_features[i]])

        return image_loss, features_loss
