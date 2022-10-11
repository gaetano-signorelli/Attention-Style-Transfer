import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.losses.mse_loss import MSELossLayer

from src.architecture.config import *

class StyleLossLayer(layers.Layer):

    def __init__(self, encoded_shape):

        super(StyleLossLayer, self).__init__()

        self.h = encoded_shape[1] #Height
        self.w = encoded_shape[2] #Width
        self.c = encoded_shape[3] #Channels

        self.mse_layer = MSELossLayer()

    @tf.function
    def get_gram_matrix(self, features, h, w, c):

        features = layers.Reshape((h*w,c))(features) #(batch_size, H*W, C)
        features_t = layers.Permute((2,1))(features) #(batch_size, C, H*W)

        gram_matrix = layers.Dot(axes=(1,2))([features, features_t]) #(batch_size, C, C)

        return gram_matrix

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
    def gram_style_loss(self, x, target, h, w, c):

        x_gram_matrix = self.get_gram_matrix(x, h, w, c)
        target_gram_matrix = self.get_gram_matrix(target, h, w, c)

        x_gram_matrix = layers.Reshape((1,c,c))(x_gram_matrix)
        target_gram_matrix = layers.Reshape((1,c,c))(target_gram_matrix)

        style_loss = self.mse_layer([x_gram_matrix, target_gram_matrix])
        style_loss = style_loss / (h*w*c)

        return style_loss

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        x_features = inputs[0]
        target_features = inputs[1]

        n_features = len(x_features)

        if USE_GRAM_STYLE_LOSS:
            loss = self.gram_style_loss(x_features[-1], target_features[-1], self.h, self.w, self.c)
            for i in range(n_features-2, -1, -1):
                exp = 2**(n_features-i-1)
                h = self.h * exp
                w = self.w * exp
                c = self.c // exp
                loss += self.gram_style_loss(x_features[i], target_features[i], h, w, c)

            #loss = loss / n_features

        else:
            loss = self.style_loss(x_features[0], target_features[0])
            for i in range(1, n_features):
                loss += self.style_loss(x_features[i], target_features[i])

        return loss
