'''
Backbones serves as a container for attributes of possible decoders.
'''

from tensorflow import keras

class Backbones:

    VGG19 = "VGG_19"
    VGG19_LIGHT = "VGG_19_Light"

    checkpoints = {
    VGG19: [0,2,5,8,13,18],
    VGG19_LIGHT: [0,2,5,8,13,18]
    }

    preprocessing_functions = {
    VGG19: keras.applications.vgg19.preprocess_input,
    VGG19_LIGHT: keras.applications.vgg19.preprocess_input
    }
