from tensorflow import keras

class Backbones:

    VGG19 = "VGG_19"

    checkpoints = {
    VGG19: [0,2,5,8,13]
    }
    
    preprocessing_functions = {
    VGG19: keras.applications.vgg19.preprocess_input
    }
