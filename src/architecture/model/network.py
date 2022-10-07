import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, Input

from src.architecture.layers.gaussian_noise_layer import GaussianNoiseLayer
from src.architecture.layers.mcc_layer import MultiChannelCorrelationLayer
from src.architecture.autoencoder.autoencoder_builder import build_autoencoder

from src.architecture.layers.losses.mse_loss import MSELossLayer
from src.architecture.layers.losses.content_loss import ContentLossLayer
from src.architecture.layers.losses.style_loss import StyleLossLayer

from src.architecture.config import *

class VSTNetwork(Model):

    def __init__(self, backbone_type, input_shape):

        super(VSTNetwork, self).__init__()

        self.encoder, self.decoder = build_autoencoder(backbone_type)
        encoded_shape = self.encoder.get_encoded_shape(Input(shape=input_shape))

        self.gaussian_noise_layer = GaussianNoiseLayer()
        self.mcc_layer = MultiChannelCorrelationLayer(encoded_shape)

        self.mse_loss_layer = MSELossLayer()
        self.content_loss_layer = ContentLossLayer()
        self.style_loss_layer = StyleLossLayer()

        self.total_loss_tracker = metrics.Mean(name="loss")
        self.content_loss_tracker = metrics.Mean(name="con")
        self.style_loss_tracker = metrics.Mean(name="sty")
        self.identity_loss_tracker = metrics.Mean(name="ide")
        self.noise_loss_tracker = metrics.Mean(name="noi")

    def set_network_weights(self, decoder_weights, mcc_weights):

        self.decoder.set_weights(decoder_weights)
        self.mcc_layer.set_weights(mcc_weights)

    def get_network_weights(self):

        return self.decoder.get_weights(), self.mcc_layer.get_weights()

    @tf.function
    def reconstruct_and_extract(self, encoded_content, encoded_style):

        mcc_stylized_content = self.mcc_layer([encoded_content, encoded_style])

        stylized_content = self.decoder(mcc_stylized_content)

        encoded_stylized_content_features = self.encoder.encode_with_checkpoints(stylized_content)
        encoded_stylized_content = encoded_stylized_content_features[-1]

        return stylized_content, encoded_stylized_content_features, encoded_stylized_content

    @tf.function
    def reconstruct(self, encoded_content, encoded_style):

        mcc_stylized_content = self.mcc_layer([encoded_content, encoded_style])

        stylized_content = self.decoder(mcc_stylized_content)

        return stylized_content

    @tf.function
    def call(self, inputs, training=False):

        assert len(inputs)==2

        content = inputs[0]
        style = inputs[1]

        encoded_content_features = self.encoder.encode_with_checkpoints(content)
        encoded_style_features = self.encoder.encode_with_checkpoints(style)

        encoded_content = encoded_content_features[-1]
        encoded_style = encoded_style_features[-1]

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
    def train_step(self, data):

        content, style = data
        inputs = [content, style]

        with tf.GradientTape() as tape:
            styl_cont, content_loss, style_loss, noise_loss, identity_loss = self(inputs, training=True)

            content_loss = content_loss * WEIGHT_CONTENT
            style_loss = style_loss * WEIGHT_STYLE
            identity_loss = identity_loss * WEIGHT_IDENTITY
            noise_loss = noise_loss * WEIGHT_NOISE

            total_loss = content_loss + style_loss + identity_loss + noise_loss

            loss = tf.math.reduce_mean(total_loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)
        self.content_loss_tracker.update_state(content_loss)
        self.style_loss_tracker.update_state(style_loss)
        self.identity_loss_tracker.update_state(identity_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "content": self.content_loss_tracker.result(),
            "style": self.style_loss_tracker.result(),
            "identity": self.identity_loss_tracker.result(),
            "noise": self.noise_loss_tracker.result(),
        }
