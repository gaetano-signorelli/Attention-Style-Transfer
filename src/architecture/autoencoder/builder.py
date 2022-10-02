from src.architecture.encoders.vgg_encoder import EncoderVGG
from src.architecture.decoders.vgg_decoder import DecoderVGG
from src.architecture.autoencoder.backbones import Backbones

def build_autoencoder(type):

    encoder = None
    decoder = None

    if type == Backbones.VGG19:
        encoder = EncoderVGG()
        decoder = DecoderVGG()

    assert encoder is not None
    assert decoder is not None

    return encoder, decoder
