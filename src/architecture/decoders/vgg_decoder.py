import tensorflow as tf
from tensorflow.keras import layers

from src.architecture.decoders.decoder_interface import Decoder
from src.architecture.autoencoder.backbones import Backbones

class DecoderVGG(Decoder):

    def __init__(self):

        super().__init__(Backbones.VGG19)

    def build_model(self):

        model = keras.Sequential(
    [
        layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu"),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.UpSampling2D(size=(2, 2), interpolation="nearest"),
        layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2DTranspose(filters=3, kernel_size=3, padding="same", activation="relu"),
    ])

        return model
