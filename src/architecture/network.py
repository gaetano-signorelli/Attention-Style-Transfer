import tensorflow as tf
from tensorflow.keras import layers, Model

from src.architecture.layers.gaussian_noise_layer import GaussianNoiseLayer
from src.architecture.layers.mcc_layer import MultiChannelCorrelationLayer
from src.architecture.autoencoder.builder import build_autoencoder

class VSTNetwork(Model):

    def __init__(self, autoencoder_type):

        super(VSTNetwork, self).__init__()

        self.encoder, self.decoder = build_autoencoder(autoencoder_type)

        self.norm_layer = layers.Normalization()
        self.gaussian_noise_layer = GaussianNoiseLayer()
        self.mcc_layer = None

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

            #Reconstructed image with style + content with noise, encoded features, last encoded features
            styl_cont_noise, enc_styl_cont_noise_feat, enc_styl_cont_noise = self.reconstruct_and_extract(encoded_content_noise, encoded_style)
            #Reconstructed image with content + content, encoded features, last encoded features
            cont_cont, enc_cont_cont_feat, enc_cont_cont = self.reconstruct_and_extract(encoded_content, encoded_content)
            #Reconstructed image with style + style, encoded features, last encoded features
            styl_styl, enc_styl_styl_feat, enc_styl_styl = self.reconstruct_and_extract(encoded_style, encoded_style)

            #TODO compute losses
            #TODO return losses
