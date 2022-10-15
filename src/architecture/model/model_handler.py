import os
import re
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from src.architecture.model.network import VSTNetwork
from src.architecture.model.lr_schedule import AdaptiveLearningRate
from src.utils.image_processing import load_preprocess_image

from src.architecture.config import *

class ModelHandler:

    def __init__(self, backbone_type, input_shape, load_model=None, verbose=True):

        self.backbone_type = backbone_type
        self.input_shape = input_shape
        self.load_model = load_model
        self.verbose = verbose

        self.model = None

        self.adapative_lr = None
        self.optimizer = None

        if self.load_model is None:
            self.load_model = LOAD_MODEL

    def build_model(self):

        if self.model is None:

            self.model = VSTNetwork(self.backbone_type, self.input_shape)

            self.initialize_model()

            if self.verbose:
                self.model.summary()

    def initialize_model(self):

        content_inputs = Input(shape=self.input_shape)
        style_inputs = Input(shape=self.input_shape)
        inputs = [content_inputs, style_inputs]

        self.model(inputs)

        current_step = 0

        if self.load_model:
            decoder_weights_path, current_step = self.get_weights()

            if decoder_weights_path is not None:
                self.load_weights(decoder_weights_path)
                if self.verbose:
                    print("Weights loaded")
                    print("Restored backup from step {}".format(current_step))

            elif self.verbose:
                print("WARNING: Weights not found: initializing model with random weights")
                print("Ignore this warning if this is the first training or a test")

        self.adapative_lr = AdaptiveLearningRate(LEARNING_RATE, LR_DECAY, current_step)
        self.optimizer = Adam(learning_rate=self.adapative_lr)

        self.model.compile(self.optimizer, run_eagerly=RUN_EAGERLY)

    def get_weights(self):

        decoder_weights = None
        current_step = 0

        pattern_decoder_weights = re.compile("decoder_\d+.npy")

        weights_files = os.listdir(WEIGHTS_PATH)
        weights_files.sort(reverse=True)

        if weights_files is not None:
            for file in weights_files:
                if pattern_decoder_weights.match(file) and decoder_weights is None:
                    decoder_weights = os.path.join(WEIGHTS_PATH, file)
                    current_step = int(decoder_weights[-10:-4])
                    break

        return decoder_weights, current_step

    def save_weights(self):

        decoder_weights = self.model.get_network_weights()

        current_step = str(self.adapative_lr.current_step).zfill(6)

        np.save(DECODER_WEIGHTS_PATH.format(current_step), decoder_weights)

        if self.verbose:
            print()
            print("Weights saved")

    def load_weights(self, decoder_weights_path):

        decoder_weights = np.load(decoder_weights_path, allow_pickle=True)

        self.model.set_network_weights(decoder_weights)

    def save_validation_results(self, index):

        validation_content = load_preprocess_image(VALIDATION_CONTENT_PATH,
                                                self.backbone_type,
                                                image_resize=IMAGE_CROP)

        validation_style = load_preprocess_image(VALIDATION_STYLE_PATH.format(index),
                                                self.backbone_type,
                                                image_resize=IMAGE_CROP)

        validation_content = np.array([validation_content])
        validation_style = np.array([validation_style])

        validation_result = self.model((validation_content, validation_style)).numpy()

        validation_result = np.clip(validation_result[0], 0, 1) *255
        validation_result = validation_result.astype(np.uint8)
        validation_result[:,:,[2,0]] = validation_result[:,:,[0,2]]

        image_result = Image.fromarray(validation_result, mode="RGB")

        image_result.save(VALIDATION_RESULT_PATH.format(index, self.adapative_lr.current_step))
