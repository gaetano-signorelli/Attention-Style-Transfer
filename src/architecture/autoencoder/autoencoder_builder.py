from src.architecture.autoencoder.encoders.vgg_encoder import EncoderVGG
from src.architecture.autoencoder.decoders.vgg_decoder import DecoderVGG
from src.architecture.autoencoder.backbones import Backbones

def build_autoencoder(type):

    encoder = None
    decoder_builder = None

    if type == Backbones.VGG19:
        encoder = EncoderVGG()
        decoder_builder = DecoderVGG()

    assert encoder is not None
    assert decoder_builder is not None

    return encoder, decoder_builder.decoder
