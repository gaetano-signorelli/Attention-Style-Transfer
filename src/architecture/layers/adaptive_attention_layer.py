'''
The following layer is the implementation of the adaptive attention mechanism
described in the official paper.
'''

import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.layers.normalize_layer import NormalizeLayer

class AdaptiveAttentionLayer(layers.Layer):

    def __init__(self, encoded_shape, last=False):

        super(AdaptiveAttentionLayer, self).__init__()

        self.h = encoded_shape[1] #Height
        self.w = encoded_shape[2] #Width
        self.c = encoded_shape[3] #Channels

        self.c1 = self.c*2 - 64 #Combined channels
        if last:
            self.c1 += self.c

        self.norm_layer = NormalizeLayer(axis=(1,2))

        self.reshape_layer_c = layers.Reshape((self.h*self.w, self.c))
        self.reshape_layer_c1 = layers.Reshape((self.h*self.w, self.c1))
        self.permute_layer = layers.Permute((2,1))

        self.conv_query = layers.Conv2D(self.c1, kernel_size=1)
        self.conv_key = layers.Conv2D(self.c1, kernel_size=1)
        self.conv_value = layers.Conv2D(self.c, kernel_size=1)

        self.softmax_layer = layers.Softmax(axis=-1)

        self.relu_layer = layers.ReLU()

    @tf.function
    def call(self, inputs):

        assert len(inputs)==4

        content = inputs[0] #(B, H, W, C)
        style = inputs[1] #(B, H, W, C)
        comb_cont = inputs[2] #(B, H, W, C1)
        comb_sty = inputs[3] #(B, H, W, C1)

        norm_content = self.norm_layer(content) #(B, H, W, C)
        norm_comb_cont = self.norm_layer(comb_cont) #(B, H, W, C1)
        norm_comb_sty = self.norm_layer(comb_sty) #(B, H, W, C1)

        Q = self.conv_query(norm_comb_cont) #(B, H, W, C1)
        K = self.conv_key(norm_comb_sty) #(B, H, W, C1)
        V = self.conv_value(style) #(B, H, W, C)

        Q_T = self.reshape_layer_c1(Q) #(B, H*W, C1)
        K = self.permute_layer(self.reshape_layer_c1(K)) #(B, C1, H*W)
        V_T = self.reshape_layer_c(V) #(B, H*W, C)

        attention_product = layers.Dot(axes=(2,1))([Q_T, K]) #(B, H*W, H*W)

        A = self.softmax_layer(attention_product) #(B, H*W, H*W)

        M = layers.Dot(axes=(2,1))([A, V_T]) #(B, H*W, C)

        M_square = tf.math.square(M) #(B, H*W, C)
        V_T_square = tf.math.square(V_T) #(B, H*W, C)

        S_square = layers.Dot(axes=(2,1))([A, V_T_square]) - M_square #(B, H*W, C)
        S_square = self.relu_layer(S_square) #(B, H*W, C)
        S_square = tf.math.maximum(S_square, 1e-9) #Necessary to avoid Nan by computing sqrt of small values
        S = tf.math.sqrt(S_square) #(B, H*W, C)

        S = layers.Reshape((self.h, self.w, self.c))(S) #(B, H, W, C)
        M = layers.Reshape((self.h, self.w, self.c))(M) #(B, H, W, C)

        output = (S * norm_content) + M #(B, H, W, C)

        return output
