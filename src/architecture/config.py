import os

from src.architecture.autoencoder.backbones import Backbones

#Backbone
BACKBONE_TYPE = Backbones.VGG19

#Loss weights
WEIGHT_CONTENT = 4
WEIGHT_STYLE = 15
WEIGHT_IDENTITY = 70
WEIGHT_NOISE = 3000

#Optimization
LEARNING_RATE = 1e-4
LR_DECAY = 5e-5

#Training
TRAIN_STEPS = 160000
BATCH_SIZE = 8

#Saving/Loading
SAVE_MODEL = True
LOAD_MODEL = True
STEPS_BEFORE_SAVE = 10000

#Paths
WEIGHTS_PATH = os.path.join("weights",BACKBONE_TYPE)
DECODER_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"decoder_{}.npy")
MCC_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"mcc_{}.npy")
