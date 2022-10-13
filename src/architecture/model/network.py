import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, Input

from src.architecture.layers.gaussian_noise_layer import GaussianNoiseLayer
from src.architecture.layers.mcc_layer import MultiChannelCorrelationLayer
from src.architecture.autoencoder.autoencoder_builder import build_autoencoder

from src.architecture.layers.losses.mse_loss import MSELossLayer
from src.architecture.layers.losses.content_loss import ContentLossLayer
from src.architecture.layers.losses.style_loss import StyleLossLayer
from src.architecture.layers.losses.identity_loss import IdentityLossLayer

from src.architecture.config import *

class VSTNetwork(Model):

    def __init__(self, backbone_type, input_shape, use_multiple_mcc):

        super(VSTNetwork, self).__init__()

        self.encoder, self.decoder, encoded_shapes = build_autoencoder(backbone_type, input_shape, use_multiple_mcc)

        self.gaussian_noise_layer = GaussianNoiseLayer()

        self.mse_loss_layer = MSELossLayer()
        self.content_loss_layer = ContentLossLayer()
        self.style_loss_layer = StyleLossLayer(encoded_shapes[-1])
        self.identity_loss_layer = IdentityLossLayer()

        self.total_loss_tracker = metrics.Mean(name="loss")
        self.content_loss_tracker = metrics.Mean(name="con")
        self.style_loss_tracker = metrics.Mean(name="sty")
        self.identity_loss_tracker = metrics.Mean(name="ide")
        self.noise_loss_tracker = metrics.Mean(name="noi")

    def set_network_weights(self, decoder_weights):

        self.decoder.set_weights(decoder_weights)

    def get_network_weights(self):

        return self.decoder.get_weights()

    @tf.function
    def reconstruct_and_extract(self, encoded_content, encoded_styles):

        stylized_content = self.decoder([encoded_content, encoded_styles])

        encoded_stylized_content_features = self.encoder.encode_with_checkpoints(stylized_content)
        encoded_stylized_content = encoded_stylized_content_features[-1]

        return stylized_content, encoded_stylized_content_features, encoded_stylized_content

    @tf.function
    def reconstruct(self, encoded_content, encoded_styles):

        stylized_content = self.decoder([encoded_content, encoded_styles])

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
        styl_cont, enc_styl_cont_feat, enc_styl_cont = self.reconstruct_and_extract(encoded_content, encoded_style_features)

        if not training:
            return styl_cont

        else:

            content_noise = self.gaussian_noise_layer(content)
            encoded_content_noise_features = self.encoder.encode_with_checkpoints(content_noise)
            encoded_content_noise = encoded_content_noise_features[-1]

            #Reconstructed image with style + content with noise
            styl_cont_noise = self.reconstruct(encoded_content_noise, encoded_style_features)
            #Reconstructed image with content + content
            cont_cont, cont_cont_feat, _ = self.reconstruct_and_extract(encoded_content, encoded_content_features)
            #Reconstructed image with style + style
            styl_styl, styl_styl_feat, _ = self.reconstruct_and_extract(encoded_style, encoded_style_features)

            content_loss = self.content_loss_layer([encoded_content, enc_styl_cont])
            style_loss = self.style_loss_layer([enc_styl_cont_feat, encoded_style_features])
            noise_loss = self.mse_loss_layer([styl_cont, styl_cont_noise])
            identity_loss_1 = self.mse_loss_layer([content, cont_cont]) + self.mse_loss_layer([style, styl_styl])
            identity_loss_2 = self.identity_loss_layer([encoded_content_features, cont_cont_feat]) + \
                            self.identity_loss_layer([encoded_style_features, styl_styl_feat])

            return styl_cont, content_loss, style_loss, noise_loss, identity_loss_1, identity_loss_2

    @tf.function
    def train_step(self, data):

        content, style = data
        inputs = [content, style]

        with tf.GradientTape() as tape:
            styl_cont, content_loss, style_loss, noise_loss, identity_loss_1, identity_loss_2 = self(inputs, training=True)

            content_loss = content_loss * WEIGHT_CONTENT
            style_loss = style_loss * WEIGHT_GRAM_STYLE if USE_GRAM_STYLE_LOSS else style_loss * WEIGHT_STYLE
            identity_loss_1 = identity_loss_1 * WEIGHT_IDENTITY_1
            identity_loss_2 = identity_loss_2 * WEIGHT_IDENTITY_2
            noise_loss = noise_loss * WEIGHT_NOISE

            total_loss = content_loss + style_loss + identity_loss_1 + identity_loss_2 + noise_loss

            loss = tf.math.reduce_mean(total_loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)
        self.content_loss_tracker.update_state(content_loss)
        self.style_loss_tracker.update_state(style_loss)
        self.identity_loss_tracker.update_state(identity_loss_1 + identity_loss_2)
        self.noise_loss_tracker.update_state(noise_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "content": self.content_loss_tracker.result(),
            "style": self.style_loss_tracker.result(),
            "identity": self.identity_loss_tracker.result(),
            "noise": self.noise_loss_tracker.result(),
        }
