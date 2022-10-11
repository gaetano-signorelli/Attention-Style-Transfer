import os

from src.architecture.autoencoder.backbones import Backbones

#Backbone
BACKBONE_TYPE = Backbones.VGG19

#Loss weights
WEIGHT_CONTENT = 4
WEIGHT_STYLE = 15
WEIGHTS_GRAM_STYLE = 0.02
WEIGHT_IDENTITY = 70
WEIGHT_NOISE = 3000

#Optimization
LEARNING_RATE = 1e-4
LR_DECAY = 5e-5

#Training
PRETRAIN = False
USE_GRAM_STYLE_LOSS = True
RUN_EAGERLY = True #False
TRAIN_STEPS = 10000 #160000
BATCH_SIZE = 8
EPOCH_LEN = 100

IMAGE_RESIZE = (512,512) #(512,512)
IMAGE_CROP = (256,256) #(256,256)

#Saving/Loading
SAVE_MODEL = True
LOAD_MODEL = False
STEPS_BEFORE_SAVE = 500 #10000

#Paths
WEIGHTS_PATH = os.path.join("weights",BACKBONE_TYPE)
DECODER_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"decoder_{}.npy")
MCC_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"mcc_{}.npy")

CONTENT_TRAIN_PATH = os.path.join("data","coco dataset")
STYLE_TRAIN_PATH = os.path.join("data","wikiart dataset")
#STYLE_TRAIN_PATH = os.path.join("data","pretrain")

VALIDATION_CONTENT_PATH = os.path.join("data","validation images", "content.jpg")
VALIDATION_STYLE_PATH = os.path.join("data","validation images", "style_{}.jpg")
VALIDATION_RESULT_PATH = os.path.join("data","validation images","validation result_{} {}.jpg")
