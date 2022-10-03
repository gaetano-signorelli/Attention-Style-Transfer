import tensorflow as tf
from tensorflow.keras import layers, metrics, Model

from src.architecture.layers.gaussian_noise_layer import GaussianNoiseLayer
from src.architecture.layers.mcc_layer import MultiChannelCorrelationLayer
from src.architecture.autoencoder.autoencoder_builder import build_autoencoder

from src.architecture.layers.losses.mse_loss import MSELossLayer
from src.architecture.layers.losses.mse_loss import ContentLossLayer
from src.architecture.layers.losses.mse_loss import StyleLossLayer

from src.architecture.config import *

class VSTNetwork(Model):

    def __init__(self, backbone_type):

        super(VSTNetwork, self).__init__()

        self.encoder, self.decoder = build_autoencoder(backbone_type)

        self.norm_layer = layers.Normalization()
        self.gaussian_noise_layer = GaussianNoiseLayer()
        self.mcc_layer = None

        self.mse_loss_layer = MSELossLayer()
        self.content_loss_layer = ContentLossLayer()
        self.style_loss_layer = StyleLossLayer()

        self.total_loss_tracker = metrics.Mean(name="loss")

    @tf.function
    def get_encoded_shapes(self, features):

        shape = tf.shape(features)

        h = tf.gather(shape, 1) #height
        w = tf.gather(shape, 2) #width
        c = tf.gather(shape, 3) #channels

        return h, w, c

    @tf.function
    def build_mcc_layer(self, encoded_content):

        if self.mcc_layer is None:
            encoded_h, encoded_w, encoded_c = self.get_encoded_shapes(encoded_content)
            self.mcc_layer = MultiChannelCorrelationLayer(encoded_h, encoded_w, encoded_c)

    @tf.function
    def reconstruct_and_extract(self, encoded_content, encoded_style):

        mcc_stylized_content = self.mcc_layer([encoded_content, encoded_style])

        stylized_content = self.decoder.decode(mcc_stylized_content)
        encoded_stylized_content_features = self.encoder.encode_with_checkpoints(stylized_content)
        encoded_stylized_content = encoded_stylized_content_features[-1]

        return stylized_content, encoded_stylized_content_features, encoded_stylized_content

    @tf.function
    def reconstruct(self, encoded_content, encoded_style):

        mcc_stylized_content = self.mcc_layer([encoded_content, encoded_style])

        stylized_content = self.decoder.decode(mcc_stylized_content)

        return stylized_content

    @tf.function
    def call(self, inputs, training=False):

        assert len(inputs)==2

        content = inputs[0]
        style = inputs[1]

        encoded_content_features = self.encoder.encode_with_checkpoints(content)
        encoded_style_features = self.encoder.encode_with_checkpoints(style)

        encoded_content = encoded_features[-1]
        encoded_style = encoded_style_features[-1]

        self.build_mcc_layer(encoded_content)

        #Reconstructed image with style + content, encoded features, last encoded features
        styl_cont, enc_styl_cont_feat, enc_styl_cont = self.reconstruct_and_extract(encoded_content, encoded_style)

        if not training:
            return styl_cont

        else:

            content_noise = self.gaussian_noise_layer(content)
            encoded_content_noise_features = self.encoder.encode_with_checkpoints(content_noise)
            encoded_content_noise = encoded_content_noise_features[-1]

            #Reconstructed image with style + content with noise
            styl_cont_noise = self.reconstruct(encoded_content_noise, encoded_style)
            #Reconstructed image with content + content
            cont_cont = self.reconstruct(encoded_content, encoded_content)
            #Reconstructed image with style + style
            styl_styl = self.reconstruct(encoded_style, encoded_style)

            content_loss = self.content_loss_layer([encoded_content, enc_styl_cont])
            style_loss = self.style_loss_layer([enc_styl_cont_feat, encoded_style_features])
            noise_loss = self.mse_loss_layer([styl_cont, styl_cont_noise])
            identity_loss = self.mse_loss_layer([content, cont_cont]) + self.mse_loss_layer([style, styl_styl])

            return styl_cont, content_loss, style_loss, noise_loss, identity_loss

    @tf.function
    def train_step(self, inputs):

        with tf.GradientTape() as tape:
            styl_cont, content_loss, style_loss, noise_loss, identity_loss = self(inputs, training=True)

            loss = content_loss * WEIGHT_CONTENT + \
                style_loss * WEIGHT_STYLE + \
                identity_loss * WEIGHT_IDENTITY + \
                noise_loss * WEIGHT_NOISE

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)

        return {
            "loss": self.total_loss_tracker.result(),
        }
