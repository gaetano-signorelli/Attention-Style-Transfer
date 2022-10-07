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

        self.model_handler.adapative_lr.update_current_step()

        if (batch+1) % STEPS_BEFORE_SAVE == 0:
            #self.model_handler.save_weights()
            self.model_handler.save_validation_results(1)
            self.model_handler.save_validation_results(2)
