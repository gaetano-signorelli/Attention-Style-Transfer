import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.normalize_layer import NormalizeLayer
from src.architecture.layers.losses.mse_loss import MSELossLayer

from src.architecture.config import *

class LocalFeatureLossLayer(layers.Layer):

    def __init__(self, encoded_shapes):

        super(LocalFeatureLossLayer, self).__init__()

        self.norm_layer = NormalizeLayer(axis=(1,2))

        self.permute_layer = layers.Permute((2,1))

        self.flatten_layers = [
        layers.Reshape((shape[1]*shape[2], shape[3]))
        for shape in encoded_shapes
        ]

        self.reshape_layers = [
        layers.Reshape((shape[1], shape[2], shape[3]))
        for shape in encoded_shapes
        ]

        self.softmax_layer = layers.Softmax(axis=-1)

        self.relu_layer = layers.ReLU()

        self.mse_layer = MSELossLayer()

    @tf.function
    def compute_aat(self, content, style, flatten_layer, reshape_layer):

        norm_content = self.norm_layer(content) #(B, H, W, C)
        norm_style = self.norm_layer(style) #(B, H, W, C)

        Q_T = flatten_layer(norm_content) #(B, H, W, C)
        K = self.permute_layer(flatten_layer(norm_style)) #(B, H, W, C)
        V_T = flatten_layer(style) #(B, H, W, C)

        attention_product = layers.Dot(axes=(2,1),normalize=True)([Q_T, K]) #(B, H*W, H*W)

        A = self.softmax_layer(attention_product) #(B, H*W, H*W)

        M = layers.Dot(axes=(2,1))([A, V_T]) #(B, H*W, C)

        M_square = tf.math.square(M) #(B, H*W, C)
        V_T_square = tf.math.square(V_T) #(B, H*W, C)

        S_square = layers.Dot(axes=(2,1))([A, V_T_square]) - M_square #(B, H*W, C)
        S_square = self.relu_layer(S_square) #(B, H*W, C)
        S = tf.math.sqrt(S_square) #(B, H*W, C)

        S = reshape_layer(S) #(B, H, W, C)
        M = reshape_layer(M) #(B, H, W, C)

        output = (S * norm_content) + M #(B, H, W, C)

        return output

    @tf.function
    def feature_loss(self, cs_feature, c_feature, s_feature, flatten_layer, reshape_layer):

        aat = self.compute_aat(c_feature, s_feature, flatten_layer, reshape_layer)

        feature_loss = self.mse_layer([cs_feature, aat])

        return feature_loss

    @tf.function
    def call(self, inputs):

        assert len(inputs)==3

        cs_features = inputs[0]
        c_features = inputs[1]
        s_features = inputs[2]

        n_features = len(cs_features)

        loss = self.feature_loss(cs_features[2], c_features[2], s_features[2],
                                self.flatten_layers[2], self.reshape_layers[2])

        for i in range(3, n_features):
            loss += self.feature_loss(cs_features[i], c_features[i], s_features[i],
                                    self.flatten_layers[i], self.reshape_layers[i])

        return loss
