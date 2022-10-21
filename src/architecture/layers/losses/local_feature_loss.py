'''
This layer computes the local feature loss as explained in the official paper.
'''

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

        self.flatten_layers_c = [
        layers.Reshape((shape[1]*shape[2], shape[3]))
        for shape in encoded_shapes
        ]

        self.flatten_layers_c1 = [
        layers.Reshape((shape[1]*shape[2], shape[3]*2 - 64))
        for shape in encoded_shapes[0:-1]
        ]
        shape = encoded_shapes[-1]
        self.flatten_layers_c1.append(layers.Reshape((shape[1]*shape[2], shape[3]*2 - 64 + shape[3])))

        self.reshape_layers = [
        layers.Reshape((shape[1], shape[2], shape[3]))
        for shape in encoded_shapes
        ]

        self.softmax_layer = layers.Softmax(axis=-1)

        self.relu_layer = layers.ReLU()

        self.mse_layer = MSELossLayer()

    @tf.function
    def compute_aat(self, content, style, comb_cont, comb_sty,
                    flatten_layer_c, flatten_layer_c1, reshape_layer):

        norm_content = self.norm_layer(content) #(B, H, W, C)
        norm_comb_cont = self.norm_layer(comb_cont) #(B, H, W, C1)
        norm_comb_sty = self.norm_layer(comb_sty) #(B, H, W, C1)

        Q_T = flatten_layer_c1(norm_comb_cont) #(B, H*W, C1)
        K = self.permute_layer(flatten_layer_c1(norm_comb_sty)) #(B, C1, H*W)
        V_T = flatten_layer_c(style) #(B, H*W, C)

        attention_product = layers.Dot(axes=(2,1))([Q_T, K]) #(B, H*W, H*W)

        A = self.softmax_layer(attention_product) #(B, H*W, H*W)

        M = layers.Dot(axes=(2,1))([A, V_T]) #(B, H*W, C)

        M_square = tf.math.square(M) #(B, H*W, C)
        V_T_square = tf.math.square(V_T) #(B, H*W, C)

        S_square = layers.Dot(axes=(2,1))([A, V_T_square]) - M_square #(B, H*W, C)
        S_square = self.relu_layer(S_square) #(B, H*W, C)
        S_square = tf.math.maximum(S_square, 1e-9) #Necessary to avoid Nan by computing sqrt of small values
        S = tf.math.sqrt(S_square) #(B, H*W, C)

        S = reshape_layer(S) #(B, H, W, C)
        M = reshape_layer(M) #(B, H, W, C)

        output = (S * norm_content) + M #(B, H, W, C)

        return output

    @tf.function
    def feature_loss(self, cs_feature, c_feature, s_feature, c_comb_feature, s_comb_feature,
                    flatten_layer_c, flatten_layer_c1, reshape_layer):

        aat = self.compute_aat(c_feature, s_feature, c_comb_feature, s_comb_feature,
                            flatten_layer_c, flatten_layer_c1, reshape_layer)

        feature_loss = self.mse_layer([cs_feature, aat])

        return feature_loss

    @tf.function
    def call(self, inputs):

        assert len(inputs)==5

        cs_features = inputs[0]
        c_features = inputs[1]
        s_features = inputs[2]
        c_comb_features = inputs[3]
        s_comb_features = inputs[4]

        n_features = len(cs_features)

        loss = self.feature_loss(cs_features[2],
                                c_features[2],
                                s_features[2],
                                c_comb_features[0],
                                s_comb_features[0],
                                self.flatten_layers_c[2],
                                self.flatten_layers_c1[2],
                                self.reshape_layers[2])

        for i in range(3, n_features):
            loss += self.feature_loss(cs_features[i],
                                    c_features[i],
                                    s_features[i],
                                    c_comb_features[i-2],
                                    s_comb_features[i-2],
                                    self.flatten_layers_c[i],
                                    self.flatten_layers_c1[i],
                                    self.reshape_layers[i])

        return loss
