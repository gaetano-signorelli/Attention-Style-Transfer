'''
This module is responsible for handling the model: its creation, the initialization,
saving and loading weights ecc.
'''

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

    def __init__(self, backbone_type, input_shape, load_model=None, weights_path=None, verbose=True):

        self.backbone_type = backbone_type
        self.input_shape = input_shape
        self.load_model = load_model
        self.weights_path = weights_path
        self.verbose = verbose

        self.model = None

        self.adapative_lr = None
        self.optimizer = None

        if self.load_model is None:
            self.load_model = LOAD_MODEL

        if self.weights_path is None:
            self.weights_path = WEIGHTS_PATH

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
            decoder_weights_path, aat_1_weights_path, aat_2_weights_path, aat_3_weights_path, current_step = self.get_weights()

            if decoder_weights_path is not None \
            or aat_1_weights_path is not None \
            or aat_2_weights_path is not None \
            or aat_3_weights_path is not None:

                self.load_weights(decoder_weights_path, aat_1_weights_path, aat_2_weights_path, aat_3_weights_path)
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

        '''
        Retrieve last saved weights and last training step
        '''

        decoder_weights = None
        aat_1_weights = None
        aat_2_weights = None
        aat_3_weights = None
        current_step = 0

        pattern_decoder_weights = re.compile("decoder_\d+.npy")
        pattern_aat_1_weights = re.compile("aat_1_\d+.npy")
        pattern_aat_2_weights = re.compile("aat_2_\d+.npy")
        pattern_aat_3_weights = re.compile("aat_3_\d+.npy")

        weights_files = os.listdir(self.weights_path)
        weights_files.sort(reverse=True)

        if weights_files is not None:
            for file in weights_files:

                if pattern_decoder_weights.match(file) and decoder_weights is None:
                    decoder_weights = os.path.join(self.weights_path, file)
                    current_step = int(decoder_weights[-10:-4])

                elif pattern_aat_1_weights.match(file) and aat_1_weights is None:
                    aat_1_weights = os.path.join(self.weights_path, file)

                elif pattern_aat_2_weights.match(file) and aat_2_weights is None:
                    aat_2_weights = os.path.join(self.weights_path, file)

                elif pattern_aat_3_weights.match(file) and aat_3_weights is None:
                    aat_3_weights = os.path.join(self.weights_path, file)

                elif decoder_weights is not None and aat_1_weights is not None \
                and aat_2_weights is not None and aat_3_weights is not None:
                    break

        return decoder_weights, aat_1_weights, aat_2_weights, aat_3_weights, current_step

    def save_weights(self):

        decoder_weights, aat_1_weights, aat_2_weights, aat_3_weights = self.model.get_network_weights()

        current_step = str(self.adapative_lr.current_step).zfill(6)

        np.save(DECODER_WEIGHTS_PATH.format(current_step), decoder_weights)
        np.save(AAT_WEIGHTS_PATH.format(1, current_step), aat_1_weights)
        np.save(AAT_WEIGHTS_PATH.format(2, current_step), aat_2_weights)
        np.save(AAT_WEIGHTS_PATH.format(3, current_step), aat_3_weights)

        if self.verbose:
            print()
            print("Weights saved")

    def load_weights(self, decoder_weights_path, aat_1_weights_path, aat_2_weights_path, aat_3_weights_path):

        decoder_weights = None
        aat_1_weights = None
        aat_2_weights = None
        aat_3_weights = None

        if decoder_weights_path is not None:
            decoder_weights = np.load(decoder_weights_path, allow_pickle=True)

        if aat_1_weights_path is not None:
            aat_1_weights = np.load(aat_1_weights_path, allow_pickle=True)
        if aat_2_weights_path is not None:
            aat_2_weights = np.load(aat_2_weights_path, allow_pickle=True)
        if aat_3_weights_path is not None:
            aat_3_weights = np.load(aat_3_weights_path, allow_pickle=True)

        self.model.set_network_weights(decoder_weights, aat_1_weights, aat_2_weights, aat_3_weights)

    def save_validation_results(self, index):

        '''
        This method keeps track of the performance of the network, by producing
        stylized contents from a small validation set.
        '''

        validation_content = load_preprocess_image(VALIDATION_CONTENT_PATH,
                                                self.backbone_type,
                                                image_resize=IMAGE_CROP)

        validation_style = load_preprocess_image(VALIDATION_STYLE_PATH.format(index),
                                                self.backbone_type,
                                                image_resize=IMAGE_CROP)

        validation_content = np.array([validation_content])
        validation_style = np.array([validation_style])

        validation_result = self.model((validation_content, validation_style)).numpy()

        validation_result = np.clip(validation_result[0], 0, 255) #Scale in range 0-255
        validation_result = validation_result.astype(np.uint8) #Cast to int
        validation_result[:,:,[2,0]] = validation_result[:,:,[0,2]] #Move from BGR to RGB

        image_result = Image.fromarray(validation_result, mode="RGB")

        image_result.save(VALIDATION_RESULT_PATH.format(index, self.adapative_lr.current_step))
