'''
This class defines the learning rate schedule to train the model.
'''

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class AdaptiveLearningRate(LearningRateSchedule):

    def __init__(self, initial_learning_rate, lr_decay, current_step):

        self.initial_learning_rate = initial_learning_rate
        self.lr_decay = lr_decay
        self.current_step = current_step

    def update_current_step(self):

        self.current_step += 1

    def __call__(self, step):

        #Scale the learning rate proportionally to the decay and the current training step
        new_learning_rate = self.initial_learning_rate / (1.0 + self.lr_decay * self.current_step)

        return new_learning_rate
