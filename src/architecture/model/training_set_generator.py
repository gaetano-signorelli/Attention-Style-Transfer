import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.architecture.autoencoder.backbones import Backbones
from src.utils.image_processing import load_preprocess_image

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

        return content_batch, style_batch

    def __len__(self):

        return self.train_steps

    def get_random_batch(self, images):

        selected_images = random.sample(images, self.batch_size)

        numpy_images = []

        for image in selected_images:
            numpy_images.append(load_preprocess_image(image, self.backbone_type, IMAGE_RESIZE, IMAGE_CROP))

        batch = np.array(numpy_images)

        return batch
