'''
Decoder based on the symmetric VGG19 network, using the adaptive attention features
as described by the paper. This decoder makes use of Separable Convolutions.
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.architecture.autoencoder.decoders.decoder_interface import Decoder
from src.architecture.autoencoder.backbones import Backbones

class DecoderVGGLight(Decoder):

    def __init__(self):

        super().__init__(Backbones.VGG19_LIGHT)

    def build_model(self):

        model = AATDecoderLight()

        return model

class AATDecoderLight(layers.Layer):

    def __init__(self):

        super(AATDecoderLight, self).__init__()

        self.up_1 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.add = layers.Add()
        self.conv_1_1 = layers.SeparableConv2D(filters=512, kernel_size=3, padding="same", activation="relu")
        self.conv_2_1 = layers.SeparableConv2D(filters=256, kernel_size=3, padding="same", activation="relu")
        self.up_2 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conc = layers.Concatenate()
        self.conv_3_1 = layers.SeparableConv2D(filters=256, kernel_size=3, padding="same", activation="relu")
        self.conv_3_2 = layers.SeparableConv2D(filters=256, kernel_size=3, padding="same", activation="relu")
        self.conv_3_3 = layers.SeparableConv2D(filters=256, kernel_size=3, padding="same", activation="relu")
        self.conv_3_4 = layers.SeparableConv2D(filters=128, kernel_size=3, padding="same", activation="relu")
        self.up_3 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv_4_1 = layers.SeparableConv2D(filters=128, kernel_size=3, padding="same", activation="relu")
        self.conv_4_2 = layers.SeparableConv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.up_4 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv_5_1 = layers.SeparableConv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.conv_5_2 = layers.SeparableConv2D(filters=3, kernel_size=3, padding="same", activation="relu")

    @tf.function
    def call(self, inputs):

        assert len(inputs)==3

        aat_1 = inputs[0]
        aat_2 = inputs[1]
        aat_3 = inputs[2]

        x = self.up_1(aat_1)
        x = self.add([x, aat_2])
        x = self.conv_1_1(x)

        x = self.conv_2_1(x)
        x = self.up_2(x)

        x = self.conc([x, aat_3])
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)
        x = self.up_3(x)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.up_4(x)

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)

        return x
