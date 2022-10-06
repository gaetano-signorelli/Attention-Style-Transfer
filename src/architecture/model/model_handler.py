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

    def __init__(self, backbone_type, input_shape, verbose=True):

        self.backbone_type = backbone_type
        self.input_shape = input_shape
        self.verbose = verbose

        self.model = None

        self.adapative_lr = None
        self.optimizer = None

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

        if LOAD_MODEL:
            decoder_weights_path, mcc_weights_path, current_step = self.get_weights()

            if decoder_weights_path is not None and mcc_weights_path is not None:
                self.load_weights(decoder_weights_path, mcc_weights_path)
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
        mcc_weights = None
        current_step = 0

        pattern_decoder_weights = re.compile("decoder_\d+.npy")
        pattern_mcc_weights = re.compile("mcc_\d+.npy")

        weights_files = os.listdir(WEIGHTS_PATH).sort(reverse=True)

        if weights_files is not None:
            for file in weights_files:

                if pattern_decoder_weights.match(file) and decoder_weights is None:
                    decoder_weights = os.path.join(WEIGHTS_PATH, file)

                elif pattern_mcc_weights.match(file) and mcc_weights is None:
                    mcc_weights = os.path.join(WEIGHTS_PATH, file)

                elif decoder_weights is not None and mcc_weights is not None:
                    decoder_current_step = int(decoder_weights[-10:-4])
                    mcc_current_step = int(mcc_weights[-10:-4])

                    if decoder_current_step!=mcc_current_step and self.verbose:
                        print("WARNING: Decoder and MCC layers have weights from different epochs")

                    current_step = min(decoder_current_step, mcc_current_step)

                    break

        return decoder_weights, mcc_weights, current_step

    def save_weights(self):

        decoder_weights, mcc_weights = self.model.get_network_weights()

        current_step = str(self.adapative_lr.current_step).zfill(6)

        np.save(DECODER_WEIGHTS_PATH.format(current_step), decoder_weights)
        np.save(MCC_WEIGHTS_PATH.format(current_step), mcc_weights)

        if self.verbose:
            print("Weights saved")

    def load_weights(self, decoder_weights_path, mcc_weights_path):

        decoder_weights = np.load(decoder_weights_path)
        mcc_weights = np.load(mcc_weights_path)

        self.model.set_network_weights(decoder_weights, mcc_weights)

    def save_validation_results(self):

        validation_content = load_preprocess_image(VALIDATION_CONTENT_PATH,
                                                self.backbone_type,
                                                image_resize=IMAGE_CROP)

        validation_style = load_preprocess_image(VALIDATION_STYLE_PATH,
                                                self.backbone_type,
                                                image_resize=IMAGE_CROP)

        validation_content = np.array([validation_content])
        validation_style = np.array([validation_style])

        validation_result = self.model((validation_content, validation_style)).numpy()

        image_result = Image.fromarray(validation_result[0], mode="RGB")

        image_result.save(VALIDATION_RESULT_PATH.format(self.adapative_lr.current_step))
