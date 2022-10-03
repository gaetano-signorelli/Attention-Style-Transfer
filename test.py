from src.architecture.autoencoder.encoders.vgg_encoder import EncoderVGG
from src.architecture.autoencoder.decoders.vgg_decoder import DecoderVGG
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

vgg_encoder = EncoderVGG()
vgg_decoder = DecoderVGG()

img = np.ones((1,256,256,3))

encoded_features = vgg_encoder.encode_with_checkpoints(img)
encoded = encoded_features[-1]
decoded = vgg_decoder.decode(encoded)

shape = tf.shape(encoded)
test_layer = layers.Conv2D(tf.gather(shape, 3), kernel_size=1)
print(test_layer(encoded))

#print(vgg_encoder.encode_with_checkpoints(img))
#print(vgg_encoder.encoder.get_layer(name = "block1_conv1").get_config()['kernel_size'])
#vgg_encoder.encoder.summary()
