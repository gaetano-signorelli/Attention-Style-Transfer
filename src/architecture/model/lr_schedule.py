from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class AdaptiveLearningRate(LearningRateSchedule):

    def __init__(self, initial_learning_rate, lr_decay, current_step):

        self.initial_learning_rate = initial_learning_rate
        self.lr_decay = lr_decay
        self.current_step = current_step

    def __call__(self, step):

        new_learning_rate = self.initial_learning_rate / (1.0 + self.lr_decay * self.current_step)

        self.current_step += 1

        return new_learning_rate
