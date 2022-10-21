'''
This module is used to preprocess images and do other manipulation on them
'''

import random
import numpy as np
from tensorflow import keras

from src.architecture.autoencoder.backbones import Backbones

def load_preprocess_image(image, backbone_type, image_resize=None, image_crop=None):

    #Load image from file in PIL format
    pil_image = keras.preprocessing.image.load_img(image,
                                                target_size=image_resize,
                                                interpolation="bilinear")

    numpy_image = np.array(pil_image)

    #Preprocess image
    if image_crop is not None:
        if image_resize is None:
            image_size = numpy_image.shape
            numpy_image = random_crop_image(numpy_image, image_size, image_crop)
        else:
            numpy_image = random_crop_image(numpy_image, image_resize, image_crop)

    #numpy_image = Backbones.preprocessing_functions[backbone_type](numpy_image)
    numpy_image[:,:,[2,0]] = numpy_image[:,:,[0,2]] #From RGB to BGR
    numpy_image = numpy_image / 1.0 #Convert to float (keep in range 0-255)

    return numpy_image

def random_crop_image(image, image_resize, image_crop):

    if image_resize==image_crop:
        return image

    max_rnd_h = image_resize[0]-1 - image_crop[0]
    max_rnd_w = image_resize[1]-1 - image_crop[1]

    assert max_rnd_h >= 0
    assert max_rnd_w >= 0

    random_h = random.randrange(max_rnd_h)
    random_w = random.randrange(max_rnd_w)

    cropped_image = image[random_h:random_h+image_crop[0], random_w:random_w+image_crop[1]]

    return cropped_image

def interpolate_images(original, target, interpolation_level):

    '''
    Take a weighted average between pixels between two images.
    Usually a content image and its stylized version.
    '''

    assert interpolation_level<=1.0 and interpolation_level>=0.0

    scaled_original = original * (1-interpolation_level)
    scaled_target = target * interpolation_level

    return scaled_original + scaled_target
