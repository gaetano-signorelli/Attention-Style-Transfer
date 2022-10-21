'''
Interface for a valid encoder.
'''

from abc import ABC, abstractmethod
import tensorflow as tf

from src.architecture.autoencoder.backbones import Backbones

class Encoder(ABC):

    def __init__(self, type):

        self.type = type

        self.encoder = self.load_model()
        self.encoder.trainable = False

        self.checkpoint_layers = self.split_by_checkpoints()

    def split_by_checkpoints(self):

        '''
        Divide encoder in groups of layers (to extract features from different levels)
        '''

        checkpoint_layers = []

        checkpoints = Backbones.checkpoints[self.type]
        layers = self.encoder.layers

        assert checkpoints[-1] <= len(layers)
        assert checkpoints[0] == 0

        for i in range(0, len(checkpoints)-1):
            start = checkpoints[i]
            end = checkpoints[i+1]

            checkpoint_layer = layers[start:end]
            checkpoint_layers.append(checkpoint_layer)

        return checkpoint_layers


    @tf.function
    def encode(self, input):
        return self.encoder(input)

    @tf.function
    def encode_with_checkpoints(self, x):

        '''
        Encode an input returning all the intermediate features
        '''

        checkpoint_results = []

        for layers in self.checkpoint_layers:

            for layer in layers:
                x = layer(x)

            checkpoint_results.append(x)

        return checkpoint_results

    def get_encoded_shapes(self, x):

        '''
        Get the shape of all the extracted (final or intermediate) features
        '''

        features = []

        for layers in self.checkpoint_layers:

            for layer in layers:
                x = layer(x)

            features.append(x)

        shapes = [feature.shape for feature in features]

        return shapes

    @abstractmethod
    def load_model(self):
        #Choose/Build the encoder's architecture
        pass
