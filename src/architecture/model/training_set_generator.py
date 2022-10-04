import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.architecture.autoencoder.backbones import Backbones

from src.architecture.config import *

class Generator(keras.utils.Sequence):

    def __init__(self, content_images, style_images, train_steps, batch_size, backbone_type):

        self.content_images = content_images
        self.style_images = style_images
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.backbone_type = backbone_type

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):

        content_batch = self.get_random_batch(self.content_images)
        style_batch = self.get_random_batch(self.style_images)

        return [content_batch, style_batch]

    def __len__(self):

        return self.train_steps

    def get_random_batch(self, images):

        selected_images = random.sample(images, self.batch_size)

        numpy_images = []

        for image in selected_images:
            numpy_images.append(self.load_preprocess_image(image))

        batch = np.array(numpy_images)

        return batch

    def load_preprocess_image(self, image):

        pil_image = keras.preprocessing.image.load_img(image,
                                                    target_size=IMAGE_RESIZE,
                                                    interpolation="bilinear")

        numpy_image = keras.preprocessing.image.img_to_array(pil_image)

        numpy_image = self.random_crop_image(numpy_image)

        numpy_image = Backbones.preprocessing_functions[self.backbone_type](numpy_image)
        print(numpy_image)

        return numpy_image

    def random_crop_image(self, image):

        max_rnd_h = IMAGE_RESIZE[0]-1 - IMAGE_CROP[0]
        max_rnd_w = IMAGE_RESIZE[1]-1 - IMAGE_CROP[1]

        assert max_rnd_h >= 0
        assert max_rnd_w >= 0

        random_h = random.randrange(max_rnd_h)
        random_w = random.randrange(max_rnd_w)

        cropped_image = image[random_h:random_h+IMAGE_CROP[0], random_w:random_w+IMAGE_CROP[1]]

        return cropped_image

    def save_partial_results(self):
        pass
