import numpy as np
from tensorflow import keras

from src.architecture.autoencoder.backbones import Backbones

def load_preprocess_image(image, backbone_type, image_resize=None, image_crop=None):

    pil_image = keras.preprocessing.image.load_img(image,
                                                target_size=image_resize,
                                                interpolation="bilinear")

    numpy_image = keras.preprocessing.image.img_to_array(pil_image)

    if image_crop is not None:
        numpy_image = self.random_crop_image(numpy_image)

    numpy_image = Backbones.preprocessing_functions[backbone_type](numpy_image)
    print(numpy_image)

    return numpy_image

def random_crop_image(image, image_resize, image_crop):

    max_rnd_h = image_resize[0]-1 - image_crop[0]
    max_rnd_w = image_resize[1]-1 - image_crop[1]

    assert max_rnd_h >= 0
    assert max_rnd_w >= 0

    random_h = random.randrange(max_rnd_h)
    random_w = random.randrange(max_rnd_w)

    cropped_image = image[random_h:random_h+image_crop[0], random_w:random_w+image_crop[1]]

    return cropped_image
