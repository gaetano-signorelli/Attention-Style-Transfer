from src.architecture.model.network import VSTNetwork

from src.architecture.config import *

class ModelHandler:

    def __init__(self, backbone_type, verbose=True):

        self.backbone_type = backbone_type
        self.verbose = verbose

        self.model = None

    def build_model(self):

        if self.model is None:

            self.model = VSTNetwork(self.backbone_type)

            self.model.compile(OPTIMIZER)

            if self.verbose:
                self.model.summary()

    def save_weights(self):
        pass

    def load_weights(self):
        pass
