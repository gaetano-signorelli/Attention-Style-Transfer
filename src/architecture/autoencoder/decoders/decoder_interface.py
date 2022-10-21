'''
Interface for a valid decoder.
'''

from abc import ABC, abstractmethod
import tensorflow as tf

class Decoder(ABC):

    def __init__(self, type):

        self.type = type

        self.decoder = self.build_model()

    @abstractmethod
    def build_model(self):
        #Choose/Build the decoder's architecture
        pass
