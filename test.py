from src.architecture.encoders.vgg_encoder import EncoderVGG
import numpy as np

vgg_encoder = EncoderVGG()
img = np.ones((1,256,256,3))
#print(vgg_encoder.encode_with_checkpoints(img))
