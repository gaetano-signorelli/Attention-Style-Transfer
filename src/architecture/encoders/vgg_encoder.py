import tensorflow as tf

from src.architecture.encoders.encoder_interface import Encoder
from src.architecture.autoencoder.backbones import Backbones

class EncoderVGG(Encoder):

    def __init__(self):

        super().__init__(Backbones.VGG19)

    def load_model(self):

        return tf.keras.applications.VGG19(include_top=False,
                                                weights="imagenet",
                                                input_tensor=None,
                                                input_shape=None,
                                                pooling=None,
                                                classes=None,
                                                classifier_activation=None)
