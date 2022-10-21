'''
VSTNetwork is the main model. It makes use of all the other separately defined
layers, embracing the entire pipeline of the style transfer.
'''

import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, Input

from src.architecture.layers.adaptive_attention_layer import AdaptiveAttentionLayer
from src.architecture.layers.feature_combination_layer import FeatureCombinatorLayer
from src.architecture.autoencoder.autoencoder_builder import build_autoencoder
from src.architecture.autoencoder.backbones import Backbones

from src.architecture.layers.losses.style_loss import StyleLossLayer
from src.architecture.layers.losses.content_loss import ContentLossLayer
from src.architecture.layers.losses.local_feature_loss import LocalFeatureLossLayer

from src.architecture.config import *

class VSTNetwork(Model):

    def __init__(self, backbone_type, input_shape):

        super(VSTNetwork, self).__init__()

        #Build encoder and decoder
        self.encoder, self.decoder,= build_autoencoder(backbone_type, input_shape)
        encoded_shapes = self.encoder.get_encoded_shapes(Input(shape=input_shape))

        self.feature_combination_layer = FeatureCombinatorLayer(encoded_shapes)

        #Define adaptive attention layers
        self.aat_1_layer = AdaptiveAttentionLayer(encoded_shapes[-1], last=True)
        self.aat_2_layer = AdaptiveAttentionLayer(encoded_shapes[-2])
        self.aat_3_layer = AdaptiveAttentionLayer(encoded_shapes[-3])

        #Define loss layers
        self.style_loss_layer = StyleLossLayer()
        self.local_feature_loss_layer = LocalFeatureLossLayer(encoded_shapes)
        self.content_loss_layer = ContentLossLayer()

        #Initialize loss metrics
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.style_loss_tracker = metrics.Mean(name="style")
        self.local_feature_loss_tracker = metrics.Mean(name="local feature")
        self.content_loss_tracker = metrics.Mean(name="content")

    def set_network_weights(self, decoder_weights=None,
                        aat_1_weights=None,
                        aat_2_weights=None,
                        aat_3_weights=None):

        '''
        Load previous state of the network
        '''

        if decoder_weights is not None:
            self.decoder.set_weights(decoder_weights)
        if aat_1_weights is not None:
            self.aat_1_layer.set_weights(aat_1_weights)
        if aat_2_weights is not None:
            self.aat_2_layer.set_weights(aat_2_weights)
        if aat_3_weights is not None:
            self.aat_3_layer.set_weights(aat_3_weights)

    def get_network_weights(self):

        '''
        Extract state of the network to save it
        '''

        decoder_weights = self.decoder.get_weights()
        aat_1_weights = self.aat_1_layer.get_weights()
        aat_2_weights = self.aat_2_layer.get_weights()
        aat_3_weights = self.aat_3_layer.get_weights()

        return decoder_weights, aat_1_weights, aat_2_weights, aat_3_weights

    @tf.function
    def call(self, inputs, training=False):

        assert len(inputs)==2

        content = inputs[0]
        style = inputs[1]

        #Encode content and style via encoder
        encoded_content_features = self.encoder.encode_with_checkpoints(content)
        encoded_style_features = self.encoder.encode_with_checkpoints(style)

        #Combine content features from previous levels
        comb_cont_1 = self.feature_combination_layer(encoded_content_features)
        comb_cont_2 = self.feature_combination_layer(encoded_content_features[0:-1])
        comb_cont_3 = self.feature_combination_layer(encoded_content_features[0:-2])

        #Combine style features from previous levels
        comb_sty_1 = self.feature_combination_layer(encoded_style_features)
        comb_sty_2 = self.feature_combination_layer(encoded_style_features[0:-1])
        comb_sty_3 = self.feature_combination_layer(encoded_style_features[0:-2])

        #Run style attention mechanism for multiple levels
        aat_1 = self.aat_1_layer([encoded_content_features[-1], encoded_style_features[-1], comb_cont_1, comb_sty_1])
        aat_2 = self.aat_2_layer([encoded_content_features[-2], encoded_style_features[-2], comb_cont_2, comb_sty_2])
        aat_3 = self.aat_3_layer([encoded_content_features[-3], encoded_style_features[-3], comb_cont_3, comb_sty_3])

        #Get stylized content via decoder
        stylized_content = self.decoder([aat_1, aat_2, aat_3])

        #Compute losses if training
        if training:

            encoded_stylized_content_features = self.encoder.encode_with_checkpoints(stylized_content)

            style_loss = self.style_loss_layer([encoded_stylized_content_features,
                                                encoded_style_features])

            c_comb_features = [comb_cont_3, comb_cont_2, comb_cont_1]
            s_comb_features = [comb_sty_3, comb_sty_2, comb_sty_1]

            local_feature_loss = self.local_feature_loss_layer([encoded_stylized_content_features,
                                                                encoded_content_features,
                                                                encoded_style_features,
                                                                c_comb_features,
                                                                s_comb_features])

            content_loss = self.content_loss_layer([encoded_content_features, encoded_stylized_content_features])

            return stylized_content, style_loss, local_feature_loss, content_loss

        else:
            return stylized_content

    @tf.function
    def train_step(self, data):

        content, style = data
        inputs = [content, style]

        with tf.GradientTape() as tape:

            stylized_content, style_loss, local_feature_loss, content_loss = self(inputs, training=True)

            #Weight losses
            style_loss = style_loss * WEIGHT_STYLE_LOSS
            local_feature_loss = local_feature_loss * WEIGHT_LOCAL_FEATURE_LOSS
            content_loss = content_loss * WEIGHT_CONTENT_LOSS

            total_loss = style_loss + local_feature_loss
            #Content loss is not used for optimization, it is only tracked

            loss = tf.math.reduce_mean(total_loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)
        self.style_loss_tracker.update_state(style_loss)
        self.local_feature_loss_tracker.update_state(local_feature_loss)
        self.content_loss_tracker.update_state(content_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "style": self.style_loss_tracker.result(),
            "local feature": self.local_feature_loss_tracker.result(),
            "content": self.content_loss_tracker.result()
        }
