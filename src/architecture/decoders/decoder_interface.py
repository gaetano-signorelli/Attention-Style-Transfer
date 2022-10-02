from abc import ABC, abstractmethod
import tensorflow as tf

class Decoder(ABC):

    def __init__(self, type):

        self.type = type

        self.decoder = self.build_model()

    @tf.function
    def decode(self, features):
        return self.decoder(features)

    @abstractmethod
    def build_model(self):
        pass
