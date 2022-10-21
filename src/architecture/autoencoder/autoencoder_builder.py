'''
This is used to build the autoencoder structure based on the selected backbone.
'''

from tensorflow.keras import Input

from src.architecture.autoencoder.encoders.vgg_encoder import EncoderVGG
from src.architecture.autoencoder.decoders.vgg_decoder import DecoderVGG
from src.architecture.autoencoder.decoders.vgg_decoder_light import DecoderVGGLight
from src.architecture.autoencoder.backbones import Backbones

def build_autoencoder(type, input_shape):

    encoder = None
    decoder_builder = None

    if type == Backbones.VGG19:
        encoder = EncoderVGG()
        decoder_builder = DecoderVGG()

    elif type == Backbones.VGG19_LIGHT:
        encoder = EncoderVGG()
        decoder_builder = DecoderVGGLight()

    assert encoder is not None
    assert decoder_builder is not None

    return encoder, decoder_builder.decoder
