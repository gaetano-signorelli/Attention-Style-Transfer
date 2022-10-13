import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.architecture.autoencoder.decoders.decoder_interface import Decoder
from src.architecture.autoencoder.backbones import Backbones
from src.architecture.layers.mcc_layer import MultiChannelCorrelationLayer
from src.architecture.layers.injection_layer import InjectionLayer

class DecoderVGG(Decoder):

    def __init__(self, encoded_shapes, use_multiple_mcc):

        self.encoded_shapes = encoded_shapes
        self.use_multiple_mcc = use_multiple_mcc

        super().__init__(Backbones.VGG19)

    def build_model(self):

        #model = keras.Sequential(
    #[
    #    layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
    #    layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
    #    layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
    #    layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
    #    layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
    #    layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu"),
    #    layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
    #    layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu"),
    #    layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"),
    #    layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
    #    layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"),
    #    layers.Conv2DTranspose(filters=3, kernel_size=3, padding="same", activation="sigmoid"),
    #],
    #name = self.type + "_decoder")

        model = MCCDecoder(self.encoded_shapes, self.use_multiple_mcc)

        return model

class MCCDecoder(layers.Layer):

    def __init__(self, encoded_shapes, use_multiple_mcc):

        super(MCCDecoder, self).__init__()

        self.encoded_shapes = encoded_shapes

        #self.mcc_1 = MultiChannelCorrelationLayer(encoded_shapes[-1])
        self.mcc_1 = InjectionLayer(encoded_shapes[-1])
        self.conv_1_1 = layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu")
        self.up_1 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv_2_1 = layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu")
        self.conv_2_2 = layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu")
        self.conv_2_3 = layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu")
        #self.mcc_2 = MultiChannelCorrelationLayer(encoded_shapes[-2]) if use_multiple_mcc else None
        self.mcc_2 = InjectionLayer(encoded_shapes[-2]) if use_multiple_mcc else None
        self.conv_2_4 = layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu")
        self.up_2 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv_3_1 = layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu")
        #self.mcc_3 = MultiChannelCorrelationLayer(encoded_shapes[-3]) if use_multiple_mcc else None
        self.mcc_3 = InjectionLayer(encoded_shapes[-3]) if use_multiple_mcc else None
        self.conv_3_2 = layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu")
        self.up_3 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv_4_1 = layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu")
        #self.mcc_4 = MultiChannelCorrelationLayer(encoded_shapes[-4]) if use_multiple_mcc else None
        self.mcc_4 = InjectionLayer(encoded_shapes[-4]) if use_multiple_mcc else None
        self.conv_4_2 = layers.Conv2DTranspose(filters=3, kernel_size=3, padding="same", activation="sigmoid")

    @tf.function
    def call(self, inputs):

        assert len(inputs)==2

        encoded_content = inputs[0]
        encoded_styles = inputs[1]

        x = self.mcc_1([encoded_content, encoded_styles[-1]])
        x = self.conv_1_1(x)
        x = self.up_1(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)
        if self.mcc_2 is not None:
            x = self.mcc_2([x, encoded_styles[-2]])
        x = self.conv_2_4(x)
        x = self.up_2(x)
        x = self.conv_3_1(x)
        #if self.mcc_3 is not None:
            #x = self.mcc_3([x, encoded_styles[-3]])
        x = self.conv_3_2(x)
        x = self.up_3(x)
        x = self.conv_4_1(x)
        if self.mcc_4 is not None:
            x = self.mcc_4([x, encoded_styles[-4]])
        x = self.conv_4_2(x)

        return x
