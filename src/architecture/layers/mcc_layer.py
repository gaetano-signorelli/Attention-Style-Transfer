import tensorflow as tf
from tensorflow.keras import layers

class MultiChannelCorrelationLayer(layers.Layer):

    def __init__(self, encoded_shape):

        super(MultiChannelCorrelationLayer, self).__init__()

        self.h = encoded_shape[1] #Height
        self.w = encoded_shape[2] #Width
        self.c = encoded_shape[3] #Channels

        #self.norm_layer = layers.Normalization(axis=(1,2))
        self.norm_layer = layers.LayerNormalization()

        self.conv_content = layers.Conv2D(self.c, kernel_size=1)
        self.conv_style = layers.Conv2D(self.c, kernel_size=1)
        self.conv_output = layers.Conv2D(self.c, kernel_size=1)

        self.dense_layer = layers.Dense(self.c)

    @tf.function
    def compute_covariance(self, x):

        x = layers.Reshape((self.w*self.h, self.c))(x) #(batch_size, W*H, C)

        x_product = tf.math.square(x) #(batch_size, W*H, C)
        x_product = tf.reduce_sum(x_product, axis=1) #(batch_size, C)

        x_sum = tf.math.reduce_sum(x, axis=1) #(batch_size, C)

        x_covariance = tf.math.divide(x_product, x_sum) #(batch_size, C)

        return x_covariance

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        content = inputs[0]
        style = inputs[1]

        content_norm = self.norm_layer(content)
        style_norm = self.norm_layer(style)

        content_norm = self.conv_content(content_norm) #(batch_size, H, W, C)
        style_norm = self.conv_style(style_norm) #(batch_size, H, W, C)

        style_covariance = self.compute_covariance(style_norm) #(batch_size, C)
        style_covariance = self.dense_layer(style_covariance) #(batch_size, C)
        style_covariance = layers.Reshape((1, 1, self.c))(style_covariance) #(batch_size, 1, 1, C)

        stylized_content = layers.Multiply()([content_norm, style_covariance]) #(batch_size, H, W, C)
        stylized_content = layers.Reshape((self.h, self.w, self.c))(stylized_content) #(batch_size, H, W, C)

        output = self.conv_output(stylized_content) #(batch_size, H, W, C)
        output = layers.Add()([output, content]) #(batch_size, H, W, C)

        return output
