from tensorflow.keras import Input

from src.architecture.autoencoder.encoders.vgg_encoder import EncoderVGG
from src.architecture.autoencoder.decoders.vgg_decoder import DecoderVGG
from src.architecture.autoencoder.backbones import Backbones

def build_autoencoder(type, input_shape, use_multiple_mcc):

    encoder = None
    decoder_builder = None
    encoded_shapes = None

    if type == Backbones.VGG19:
        encoder = EncoderVGG()
        encoded_shapes = encoder.get_encoded_shapes(Input(shape=input_shape))
        decoder_builder = DecoderVGG(encoded_shapes, use_multiple_mcc)

    assert encoder is not None
    assert decoder_builder is not None
    assert encoded_shapes is not None

    return encoder, decoder_builder.decoder, encoded_shapes
