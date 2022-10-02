import tensorflow as tf
from tensorflow.keras import layers

class MultiChannelCorrelationLayer(layers.Layer):

    def __init__(self, input_filters, input_w, input_h):

        self.c = input_filters #Channels
        self.w = input_w #Width
        self.h = input_h #Height

        self.norm_layer = layers.Normalization()

        self.conv_content = layers.Conv2D(input_filters, kernel_size=1)
        self.conv_style = layers.Conv2D(input_filters, kernel_size=1)
        self.conv_output = layers.Conv2D(input_filters, kernel_size=1)

        self.dense_layer = layers.Dense(input_filters)

    @tf.function
    def compute_covariance(self, x):

        x = layers.Reshape((self.w*self.h, self.c))(x) #(batch_size, W*H, C)

        x_product = layers.Dot(axes=(1))([x, x]) # (batch_size, 1, C)
        reshaped_x_product = layers.Reshape((self.c))(x_product) # (batch_size, C)

        x_sum = tf.math.reduce_sum(x, axis=1) #(batch_size, C)

        x_covariance = tf.math.divide(x_product, x_sum) #(batch_size, C)

        return x_covariance

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        content = inputs[0]
        style = inputs[1]

        content = self.norm_layer(content)
        style = self.norm_layer(style)

        content = self.conv_content(content) #(batch_size, H, W, C)
        style = self.conv_style(style) #(batch_size, H, W, C)

        style_covariance = self.compute_covariance(style) #(batch_size, C)
        style_covariance = self.dense_layer(style_covariance) #(batch_size, C)
        style_covariance = layers.Reshape((1, 1, self.c))(style_covariance) #(batch_size, 1, 1, C)

        stylized_content = layers.Multiply()([content, style_covariance]) #(batch_size, H, W, C)
        stylized_content = layers.Reshape((self.h, self.w, self.c))(stylized_content) #(batch_size, H, W, C)

        output = self.conv_output(stylized_content) #(batch_size, H, W, C)
        output = layers.Add()[output, content]

        return output
