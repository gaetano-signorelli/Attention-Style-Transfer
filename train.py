'''
This script runs a train session.
'''

import os

from src.architecture.model.callbacks import SaveUpdateStepCallback
from src.architecture.model.model_handler import ModelHandler
from src.architecture.model.training_set_generator import Generator

from src.architecture.config import *

def get_filenames(path):

    files = os.listdir(path)

    filenames = [os.path.join(path, file) for file in files]

    return filenames

if __name__ == '__main__':

    input_shape = (IMAGE_CROP[0],IMAGE_CROP[1],3)

    #Load model
    model_handler = ModelHandler(BACKBONE_TYPE, input_shape)
    model_handler.build_model()

    callback = SaveUpdateStepCallback(model_handler)

    #Load datasets
    content_images = get_filenames(CONTENT_TRAIN_PATH)
    style_images = get_filenames(STYLE_TRAIN_PATH)

    remaining_train_steps = TRAIN_STEPS - model_handler.adapative_lr.current_step
    epochs = remaining_train_steps // EPOCH_LEN

    generator = Generator(content_images, style_images, BATCH_SIZE, BACKBONE_TYPE)

    #Run the training
    model_handler.model.fit(generator, callbacks=[callback], epochs=epochs)
