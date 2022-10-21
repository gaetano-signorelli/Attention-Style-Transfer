'''
This module contains all the constants and the parameters used at training time.
Set the new values before starting a new training session, if required.
'''

import os

from src.architecture.autoencoder.backbones import Backbones

#Backbone
BACKBONE_TYPE = Backbones.VGG19

#Loss weights
WEIGHT_STYLE_LOSS = 10
WEIGHT_LOCAL_FEATURE_LOSS = 3
WEIGHT_CONTENT_LOSS = 4

#Optimization
LEARNING_RATE = 1e-4
LR_DECAY = 5e-5

#Training
RUN_EAGERLY = True #False
TRAIN_STEPS = 40000 #50000
BATCH_SIZE = 8
EPOCH_LEN = 100

IMAGE_RESIZE = (512,512)
IMAGE_CROP = (256,256)

#Saving/Loading
SAVE_MODEL = True
LOAD_MODEL = True
STEPS_BEFORE_SAVE = 1000

#Paths
WEIGHTS_PATH = os.path.join("weights",BACKBONE_TYPE)
DECODER_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"decoder_{}.npy")
AAT_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"aat_{}_{}.npy")

CONTENT_TRAIN_PATH = os.path.join("data","coco dataset")
STYLE_TRAIN_PATH = os.path.join("data","wikiart dataset")

VALIDATION_CONTENT_PATH = os.path.join("data","validation images", "content.jpg")
VALIDATION_STYLE_PATH = os.path.join("data","validation images", "style_{}.jpg")
VALIDATION_RESULT_PATH = os.path.join("data","validation images","validation result_{} {}.jpg")
