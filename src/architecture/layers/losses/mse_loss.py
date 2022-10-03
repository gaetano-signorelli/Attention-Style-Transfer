import tensorflow as tf
from tensorflow.keras import layers

class MSELossLayer(layers.Layer):

    def __init__(self):

        super(MSELossLayer, self).__init__()

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        x = inputs[1]
        target = inputs[2]

        error = x - target
        square_error = tf.math.square(error)
        loss = tf.math.reduce_mean(square_error, axis=(1,2,3))

        return loss
