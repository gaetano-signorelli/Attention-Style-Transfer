'''
This module is a container for all the callbacks class called by the network.
'''

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from src.architecture.model.lr_schedule import AdaptiveLearningRate
from src.architecture.model.model_handler import ModelHandler

from src.architecture.config import *

class SaveUpdateStepCallback(Callback):

    def __init__(self, model_handler):

        super(SaveUpdateStepCallback, self).__init__()

        self.model_handler = model_handler

    def on_train_batch_end(self, batch, logs=None):

        '''
        Update current step in learning rate schedule and save weights and compute
        validation images after the specified number of steps.
        '''

        self.model_handler.adapative_lr.update_current_step()

        if self.model_handler.adapative_lr.current_step % STEPS_BEFORE_SAVE == 0:

            if SAVE_MODEL:
                self.model_handler.save_weights()

            self.model_handler.save_validation_results(1)
            self.model_handler.save_validation_results(2)
            self.model_handler.save_validation_results(3)
