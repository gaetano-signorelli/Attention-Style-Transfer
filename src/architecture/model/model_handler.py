import os
import re
import numpy as np
from tensorflow.keras.optimizers import Adam

from src.architecture.model.network import VSTNetwork
from src.architecture.model.lr_schedule import AdaptiveLearningRate

from src.architecture.config import *

class ModelHandler:

    def __init__(self, backbone_type, verbose=True):

        self.backbone_type = backbone_type
        self.verbose = verbose

        self.model = None

        self.adapative_lr = None
        self.optimizer = None

    def build_model(self):

        if self.model is None:

            self.model = VSTNetwork(self.backbone_type)

            self.initialize_model()

            if self.verbose:
                self.model.summary()

    def initialize_model(self):

        if LOAD_MODEL:
            decoder_weights_path, mcc_weights_path, current_step = self.get_weights()

            if decoder_weights_path is not None and mcc_weights_path is not None:
                self.load_weights(decoder_weights_path, mcc_weights_path)
                if self.verbose:
                    print("Weights loaded")

            elif self.verbose:
                print("WARNING: Weights not found: initializing model with random weights")
                print("Ignore this warning if this is the first training or a test")

        self.adapative_lr = AdaptiveLearningRate(LEARNING_RATE, LR_DECAY, current_step)
        self.optimizer = Adam(learning_rate=adapative_lr)

        self.model.compile(self.optimizer)

    def get_weights(self):

        decoder_weights = None
        mcc_weights = None
        current_step = 0

        pattern_decoder_weights = re.compile("decoder_\d+.npy")
        pattern_mcc_weights = re.compile("mcc_\d+.npy")

        weights_files = os.listdir(WEIGHTS_PATH).sort(reverse=True)

        for file in weights_files:

            if pattern_decoder_weights.match(file) and decoder_weights is None:
                decoder_weights = file

            elif pattern_mcc_weights.match(file) and mcc_weights is None:
                mcc_weights = file

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
